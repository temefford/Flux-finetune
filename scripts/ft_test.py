import argparse
import math
import os
import time
from functools import partial
from pathlib import Path

import datasets
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.data
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers import FluxPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from peft import LoraConfig, PeftModel
from PIL import Image
from tqdm.auto import tqdm

import numpy as np # Import numpy for type checking

logger = get_logger(__name__)

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a FLUX training script.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ft_config.yaml",
        help="Path to the configuration YAML file relative to the project root.",
    )
    # Add arguments for potential overrides
    parser.add_argument("--data_dir", type=str, default=None, help="Override dataset path from config.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory from config.")
    parser.add_argument("--validation_split", type=float, default=None, help="Validation split ratio (overrides config).")
    parser.add_argument("--log_level", type=str, default=None, help="Logging level (overrides config).") # Added for consistency
    parser.add_argument(
        "--caption_column", type=str, default="text", help="The name of the caption column in the dataset."
    )
    parser.add_argument(
        "--hash_column", type=str, default="hash", help="The name of the hash column in the dataset (used as dummy text input)."
    )

    cmd_args = parser.parse_args() # Parse command line args first

    # --- Load Configuration from YAML --- #
    try:
        with open(cmd_args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at: {cmd_args.config}") # Use print
        raise
    except Exception as e:
        print(f"ERROR: Error loading configuration file {cmd_args.config}: {e}") # Use print
        raise

    # Create Namespace from YAML config first
    args = argparse.Namespace(**config)
    # Store the path to the config file itself if needed later
    args.config_path = cmd_args.config 

    # Ensure dataset_path is populated from data_dir in config if it exists
    if hasattr(args, 'data_dir') and not hasattr(args, 'dataset_path'):
        args.dataset_path = args.data_dir
        print(f"INFO: Using data_dir '{args.data_dir}' from config as dataset_path.") # Use print

    # Override specific keys if they were provided via command line
    if cmd_args.data_dir:
        args.dataset_path = cmd_args.data_dir # Override whatever was set from config
        print(f"INFO: Overriding dataset_path with command-line value: {args.dataset_path}") # Use print
    if cmd_args.output_dir:
        args.output_dir = cmd_args.output_dir
        print(f"INFO: Overriding output_dir with command-line value: {args.output_dir}") # Use print
    if cmd_args.validation_split is not None: # Check if explicitly provided
        args.validation_split = cmd_args.validation_split
        print(f"INFO: Overriding validation_split with command-line value: {args.validation_split}") # Use print
    if cmd_args.log_level:
        args.log_level = cmd_args.log_level
        print(f"INFO: Overriding log_level with command-line value: {args.log_level}") # Use print

    # --- Ensure essential args exist and perform adjustments --- #
    # Validate required args that might not have defaults
    if not hasattr(args, 'dataset_path') or not args.dataset_path:
        raise ValueError("dataset_path must be specified in the config file (as data_dir or dataset_path) or via --data_dir")
    if not hasattr(args, 'output_dir') or not args.output_dir:
        raise ValueError("output_dir must be specified in the config file or via --output_dir")

    # Make output_dir absolute if it's relative
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Add other defaults if missing from config or command line
    args.config_path = cmd_args.config # Store the actual config path used
    args.validation_batch_size = getattr(args, 'validation_batch_size', args.batch_size)
    args.dataloader_num_workers = getattr(args, 'dataloader_num_workers', 4)
    args.preprocessing_num_workers = getattr(args, 'preprocessing_num_workers', 1)
    # Ensure validation_split has a default value if not set anywhere
    if not hasattr(args, 'validation_split'):
        args.validation_split = 0.0 # Default to 0 if not in config or cmd line
        print("WARNING: validation_split not found in config or command line, defaulting to 0.0") # Use print

    return args

# --- Data Preprocessing Helper Functions ---
def preprocess_train(examples, dataset_abs_path, image_transforms, image_column, hash_column, caption_column, tokenizer_2):
    """Preprocesses a batch using batched processor/tokenizer calls, returning batch tensors."""
    image_paths = [os.path.join(dataset_abs_path, f"{fn}.jpg") for fn in examples[image_column]]
    batch_size = len(image_paths)

    try:
        # --- Load Images Individually with Error Handling --- #
        images = []
        valid_image_paths = [] # Keep track of paths for images loaded successfully
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_image_paths.append(path)
            except IndexError as ie:
                logger.error(f"IndexError loading/converting image: {path}. Error: {ie}. Appending None.")
                images.append(None) # Append None if specific image fails
            except FileNotFoundError:
                 logger.warning(f"Image file not found during preprocessing: {path}. Appending None.")
                 images.append(None)
            except Exception as img_err:
                 logger.warning(f"Error loading/converting image {path}: {img_err}. Appending None.")
                 images.append(None)

        # Filter out None entries before passing to processor
        valid_images = [img for img in images if img is not None]

        if not valid_images:
            logger.error("No valid images could be loaded in this batch.")
            return {"pixel_values": None, "input_ids_2": None}

        # --- Image Processing (Batched using image_processor.preprocess) ---
        # Log the paths *intended* for this batch (even if some failed loading)
        logger.info(f"Attempting to preprocess batch for image paths: {image_paths}")
        # Log the paths for images that were *successfully* loaded and are being processed
        logger.info(f"Processing valid image paths: {valid_image_paths}")

        # image_transforms is now pipeline.image_processor.preprocess
        # Pass only the successfully loaded images
        image_inputs = image_transforms(valid_images)
        pixel_values_maybe_numpy = image_inputs['pixel_values'] # Might be numpy array

        # Convert to tensor if it's a numpy array
        if isinstance(pixel_values_maybe_numpy, np.ndarray):
            pixel_values_batch_tensor = torch.from_numpy(pixel_values_maybe_numpy)
        elif isinstance(pixel_values_maybe_numpy, torch.Tensor):
            pixel_values_batch_tensor = pixel_values_maybe_numpy # Already a tensor
        else:
             logger.error(f"Unexpected type from VaeImageProcessor: {type(pixel_values_maybe_numpy)}. Expected numpy array or tensor.")
             # Handle error: return None to be filtered by collate_fn
             return {"pixel_values": None, "input_ids_2": None}

        # --- Text Processing (Batched) ---
        # IMPORTANT: Need to ensure text corresponds to *valid* images
        # We need the original indices of the valid images to select the correct captions
        valid_indices = [i for i, img in enumerate(images) if img is not None]
        if len(valid_indices) != len(valid_images):
             logger.error("Mismatch between valid_images and valid_indices counts.")
             # Handle error case - maybe return None for batch
             return {"pixel_values": None, "input_ids_2": None}

        # Select captions corresponding to the valid images
        valid_captions = [str(examples[caption_column][i]) if examples[caption_column][i] is not None else "" for i in valid_indices]

        # Process only the valid captions
        max_len = getattr(tokenizer_2, 'model_max_length', 512)
        if not valid_captions:
             logger.error("No valid captions found corresponding to valid images.")
             return {"pixel_values": None, "input_ids_2": None} # Should align with no valid images case

        text_inputs = tokenizer_2(
            valid_captions, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt"
        )
        input_ids_2_batch_tensor = text_inputs['input_ids'] # Tensor [NumValid, SeqLen]

        # --- Sanity Check Batch Sizes (Should match num valid images) ---
        num_valid = len(valid_images)
        if pixel_values_batch_tensor.shape[0] != num_valid or input_ids_2_batch_tensor.shape[0] != num_valid:
            logger.error(f"Batch size mismatch after processing valid items in preprocess_train. Expected {num_valid}, got {pixel_values_batch_tensor.shape[0]} images and {input_ids_2_batch_tensor.shape[0]} texts.")
            return {
                "pixel_values": None,
                "input_ids_2": None,
            }

        # --- Return Batch Tensors ---
        # Note: The dataset.map function expects the output dict keys to align with dataset columns.
        # However, the tensors we return now correspond only to the *valid* examples in the input batch.
        # This mismatch might cause issues if dataset.map strictly requires output lists/tensors
        # to have the same length as the input batch size.
        # A potential solution is to pad the output tensors with dummy data for the failed examples,
        # or structure the return differently if map allows it. Let's try returning directly first.
        # If map complains, we might need to return lists of tensors/Nones again.
        return {
            "pixel_values": pixel_values_batch_tensor, # Tensor for valid images
            "input_ids_2": input_ids_2_batch_tensor,   # Tensor for valid captions
        }

    except FileNotFoundError as e: # This might be redundant now
        logger.error(f"Error opening image file: {e}. Returning None for batch.")
        return {"pixel_values": None, "input_ids_2": None}
    except Exception as e:
        logger.error(f"General error during preprocessing batch (after image loading): {e}")
        return {"pixel_values": None, "input_ids_2": None}

def preprocess_imagefolder(examples, image_transforms, image_column):
    """Preprocesses imagefolder batch using image_processor.preprocess."""
    batch_size = len(examples[image_column])
    try:
        images = [image.convert("RGB") for image in examples[image_column]]

        # Process image batch using image_processor.preprocess
        image_inputs = image_transforms(images, return_tensors="pt")
        pixel_values_batch_tensor = image_inputs['pixel_values']

        # Create dummy text inputs (batch tensor)
        # Note: Ensure tokenizer_2 exists or define max_len
        # max_len = 77
        # dummy_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        dummy_ids = torch.zeros((batch_size, 1), dtype=torch.long) # Simpler placeholder

        if pixel_values_batch_tensor.shape[0] != batch_size:
             logger.error(f"Batch size mismatch for images in preprocess_imagefolder. Expected {batch_size}, got {pixel_values_batch_tensor.shape[0]}.")
             return {"pixel_values": None, "input_ids_2": None}

        return {
            "pixel_values": pixel_values_batch_tensor,
            "input_ids_2": dummy_ids,
        }
    except Exception as e:
        logger.error(f"Error during imagefolder preprocessing batch: {e}")
        return {"pixel_values": None, "input_ids_2": None}

# --- Main Function ---
def main(args):
    # Initialize Accelerator
    logging_dir = Path(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard", # or "wandb" if configured
        project_config=accelerator_project_config,
    )

    accelerator.print(f"DEBUG: Effective args.data_dir after parsing: {args.data_dir}")

    # Make one log on every process with the configuration for debugging.
    accelerator.print(f"Accelerator state: {accelerator.state}")

    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Determine weight dtype based on mixed precision setting
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32 # Default to float32 if 'no' or unspecified
    logger.info(f"Using weight dtype: {weight_dtype}")

    # --- Load Models and Tokenizers ---
    try:
        # Load scheduler, tokenizer and models.
        # FLUX uses a specific pipeline structure
        logger.info("Loading base model pipeline...")
        pipeline = FluxPipeline.from_pretrained(
            args.model_id, # Use model_id from config
            revision=getattr(args, 'revision', None), # Safely get revision, default to None
            torch_dtype=weight_dtype, # Use weight_dtype here during initial load
        )
        logger.info("Pipeline loaded.")

        # Extract components
        original_scheduler_config = pipeline.scheduler.config
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=original_scheduler_config.get("num_train_timesteps", 1000), 
            beta_schedule=original_scheduler_config.get("beta_schedule", "scaled_linear"),
            prediction_type=original_scheduler_config.get("prediction_type", "epsilon") # Common default for training
        )
        logger.info(f"Initialized training noise scheduler: {noise_scheduler.__class__.__name__}")
        
        vae = pipeline.vae
        text_encoder = pipeline.text_encoder
        text_encoder_2 = pipeline.text_encoder_2
        transformer = pipeline.transformer # This is the main model to fine-tune
        logger.info("Model components extracted.")

        # Move VAE to device and cast dtype
        vae.to(accelerator.device, dtype=weight_dtype)
        logger.info(f"VAE moved to device {accelerator.device} and cast to {weight_dtype}")

        # Log channel configurations to debug potential mismatch
        logger.info(f"Transformer class: {transformer.__class__.__name__}")
        logger.info(f"VAE configured latent channels: {vae.config.latent_channels}")
        logger.info(f"Transformer configured input channels: {transformer.config.in_channels}")
        logger.info(f"Transformer cross_attention_dim: {getattr(transformer.config, 'cross_attention_dim', 'N/A')}") # Added logging
        logger.info("Transformer x_embedder weight shape: N/A") # Placeholder, adjust if needed

        tokenizer = pipeline.tokenizer
        tokenizer_2 = pipeline.tokenizer_2
        logger.info("Tokenizers extracted.")

    except Exception as e:
        logger.error(f"Failed to load models or components: {e}")
        return

    # Freeze VAE and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    transformer.requires_grad_(False) # Start with transformer frozen, LoRA will unfreeze target modules

    # --- Add LoRA to transformer (UNet equivalent) ---
    if args.peft_method == "LoRA":
        logger.info("Adding LoRA layers to the transformer (UNet equivalent).")
        transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank, # Often set equal to rank
            # Target the Q, K, V projections based on printed names
            target_modules=["to_q", "to_k", "to_v"], # Corrected target modules
            lora_dropout=0.1, # Optional dropout
            bias="none",
        )
        transformer.add_adapter(transformer_lora_config)
        # Ensure LoRA layers are float32 for stability if using mixed precision
        if args.mixed_precision == "fp16":
             for name, param in transformer.named_parameters():
                if "lora_" in name:
                    param.data = param.data.to(torch.float32)
        logger.info(f"Added LoRA with rank {args.lora_rank} to {transformer_lora_config.target_modules}")

    else:
        logger.warning(f"PEFT method '{args.peft_method}' not implemented for this script. Training full model.")
        transformer.requires_grad_(True) # Train full transformer if not LoRA

    # --- Define Projection Layer for VAE->Transformer Channel Mismatch --- 
    # Based on confirmed config values and runtime checks
    vae_latent_channels_actual = vae.config.latent_channels # Should be 16
    transformer_in_channels_actual = transformer.config.in_channels # Should be 64
    vae_to_transformer_projection = None # Initialize as None
    params_to_optimize = list(transformer.parameters()) # Start with transformer params (LoRA or full)

    if vae_latent_channels_actual != transformer_in_channels_actual:
        projection_method = getattr(args, 'vae_projection_method', 'conv').lower()
        logger.warning(
            f"Runtime shape mismatch detected: VAE output {vae_latent_channels_actual} channels, "
            f"Transformer input {transformer_in_channels_actual} channels. "
            f"Adding '{projection_method}' projection layer {vae_latent_channels_actual}->{transformer_in_channels_actual}."
        )
        
        if projection_method == 'conv':
            # Use Conv2d for spatial data (B, C_in, H, W) -> (B, C_out, H, W)
            vae_to_transformer_projection = torch.nn.Conv2d(
                vae_latent_channels_actual,
                transformer_in_channels_actual,
                kernel_size=1 # 1x1 convolution acts like a per-pixel linear layer across channels
            )
        elif projection_method == 'linear':
            # Use Linear - WARNING: This likely requires reshaping latents BEFORE projection
            # Current code applies projection before reshaping, suitable for Conv2d
            logger.warning("Using 'linear' projection. Ensure latents are reshaped before this layer if needed.")
            vae_to_transformer_projection = torch.nn.Linear(vae_latent_channels_actual, transformer_in_channels_actual)
        else:
            raise ValueError(f"Unsupported vae_projection_method: {projection_method}. Choose 'conv' or 'linear'.")

        # Move projection layer to device/dtype and add its parameters for optimization
        vae_to_transformer_projection.to(accelerator.device, dtype=weight_dtype)
        params_to_optimize.extend(vae_to_transformer_projection.parameters())
        logger.info("Added VAE->Transformer projection layer parameters to optimizer.")

    # If no projection needed, params_to_optimize just contains transformer parameters
    # --- Optimizer Setup --- 
    # Use AdamW optimizer (common for transformer models)
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=float(args.learning_rate), # Ensure learning_rate is float
        betas=(0.9, 0.999),
        weight_decay=1e-2, # Common weight decay
        eps=1e-08,
    )

    # --- Dataset Loading and Preprocessing ---
    # Calculate absolute dataset path relative to the current working directory
    dataset_abs_path = os.path.abspath(args.dataset_path)

    logger.info(f"Loading dataset. Type: {args.dataset_type}, Path: {dataset_abs_path}")

    # --- Load based on type ---
    if args.dataset_type == "imagefolder":
        dataset = datasets.load_dataset(
            "imagefolder",
            data_dir=dataset_abs_path,
            split="train",
        )
        logger.info(f"Loaded dataset with {len(dataset)} examples.")
    elif args.dataset_type == "hf_metadata":
        metadata_path = os.path.join(dataset_abs_path, "metadata.json")
        logger.info(f"Checking for metadata file for 'hf_metadata' type: {metadata_path}")
        if not os.path.exists(metadata_path):
            logger.error(f"Metadata file absolutely required but not found at {metadata_path}")
            logger.error("Exiting. Ensure 'metadata.json' exists in the specified dataset_path for 'hf_metadata' type.")
            return # Stop execution
        logger.info("Metadata file found, proceeding with loading.")
        # Load using metadata.json
        try:
            dataset = load_dataset("json", data_files=metadata_path, split="train") # Removed field="data"
            logger.info(f"Loaded dataset with {len(dataset)} examples.")
        except Exception as e:
            logger.error(f"Failed to load dataset from {metadata_path}: {e}")
            logger.error("Ensure 'metadata.json' exists and is formatted correctly (JSON array of objects with 'file_name' and 'text').")
            logger.error("Alternatively, modify the 'load_dataset' call for your structure.")
            return
    else:
        raise ValueError("Unsupported dataset type. Please use 'imagefolder' or 'hf_metadata'.")

    # --- Split Dataset if validation_split is provided --- #
    if args.validation_split > 0.0: # <-- Correctly uses validation_split
        split_seed = getattr(args, 'seed', 42) # Use main seed if available
        full_dataset = dataset # Assign the loaded dataset before splitting
        # Use the datasets library's built-in splitting method
        split_dataset = full_dataset.train_test_split(test_size=args.validation_split, seed=split_seed) # <-- Correctly uses validation_split
        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]
        logger.info(f"  Training samples: {len(train_dataset)}")
        logger.info(f"  Validation samples: {len(val_dataset)}")
    else:
        # Use the full dataset for training if no validation split
        train_dataset = dataset
        val_dataset = None
        logger.info(f"Using full dataset for training ({len(train_dataset)} samples). No validation split.")

    # Prepare the dataset
    logger.info("Loading dataset...")
    if args.dataset_type == "hf_metadata":
        # Ensure dataset_path is the directory containing metadata.jsonl
        data_files = {"train": os.path.join(args.dataset_path, "metadata.json")}
        dataset = load_dataset("json", data_files=data_files, split="train")
        # Filter dataset if filter_field and filter_value are provided
        if getattr(args, 'filter_field', None) and getattr(args, 'filter_value', None):
            logger.info(f"Filtering dataset: {getattr(args, 'filter_field', None)} == '{getattr(args, 'filter_value', None)}'")
            dataset = dataset.filter(lambda example: example.get(getattr(args, 'filter_field', None)) == getattr(args, 'filter_value', None))
            logger.info(f"Filtered dataset size: {len(dataset)}")
        dataset = dataset.shuffle(seed=args.seed)

        # Determine columns to remove
        original_columns = dataset.column_names
        # Keep necessary columns: image, caption, hash (maybe others based on config)
        # Columns needed by preprocess_train: image_column, hash_column, caption_column
        columns_to_keep = {args.image_column, args.caption_column} # Start with essential ones
        # Add hash_column only if it exists and is different from image_column (optional)
        hash_column_val = getattr(args, 'hash_column', None)
        if hash_column_val: # Only add if defined in args
             columns_to_keep.add(hash_column_val)
        columns_to_remove = [col for col in original_columns if col not in columns_to_keep]
        logger.info(f"Columns to keep: {columns_to_keep}")
        logger.info(f"Columns to remove: {columns_to_remove}")

        # --- Prepare Preprocessing Function --- #
        _preprocess_train_func = partial(
            preprocess_train,
            dataset_abs_path=args.dataset_path,
            # Pass the image processor's preprocess method directly
            image_transforms=pipeline.image_processor.preprocess,
            image_column=args.image_column,
            hash_column=getattr(args, 'hash_column', None), # Pass hash_column if defined, else None
            caption_column=args.caption_column,
            tokenizer_2=tokenizer_2,
        )

        logger.info("Preprocessing dataset...")
        processed_dataset = dataset.map(
            _preprocess_train_func,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=columns_to_remove,
            desc="Running tokenizer on train dataset",
        )

    elif args.dataset_type == "imagefolder":
        dataset = load_dataset("imagefolder", data_dir=args.dataset_path, split="train")
        dataset = dataset.shuffle(seed=args.seed)
        original_columns = dataset.column_names
        columns_to_keep = {args.image_column} # Only need image for imagefolder
        columns_to_remove = [col for col in original_columns if col not in columns_to_keep]
        logger.info(f"Columns to remove: {columns_to_remove}")

        _preprocess_imagefolder_func = partial(
             preprocess_imagefolder,
             # Pass the image processor's preprocess method directly
             image_transforms=pipeline.image_processor.preprocess,
             image_column=args.image_column,
        )
        processed_dataset = dataset.map(
             _preprocess_imagefolder_func,
             batched=True,
             num_proc=args.preprocessing_num_workers,
             remove_columns=columns_to_remove,
             desc="Running preprocessing on imagefolder dataset",
        )
    else:
        raise ValueError(f"Unsupported dataset_type: {args.dataset_type}")

    # --- Split Dataset --- #
    if args.validation_split > 0.0:
        split_seed = getattr(args, 'split_seed', args.seed)
        logger.info(f"Splitting dataset with validation split {args.validation_split} and seed {split_seed}")
        split_dataset = processed_dataset.train_test_split(test_size=args.validation_split, seed=split_seed)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        logger.info(f"Train dataset size: {len(train_dataset)}, Eval dataset size: {len(eval_dataset)}")
    else:
        train_dataset = processed_dataset
        eval_dataset = None
        logger.info(f"Train dataset size: {len(train_dataset)}")

    # --- Create DataLoader --- #
    logger.info("Creating DataLoader...")

    # Collate function (Revert to simplest form)
    def collate_fn(examples):
        # Filter out examples where preprocessing might have failed (returned None)
        valid_examples = [ex for ex in examples if ex.get("pixel_values") is not None and ex.get("input_ids_2") is not None]

        if not valid_examples:
            # logger.warning("Collate fn: No valid examples found after filtering Nones.")
            return {} # Return empty batch if all examples failed

        # If map unpacked correctly, examples should have single tensors
        try:
            pixel_values = torch.stack([example["pixel_values"] for example in valid_examples])
            input_ids_2 = torch.stack([example["input_ids_2"] for example in valid_examples])

            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

            return {
                "pixel_values": pixel_values,
                "input_ids_2": input_ids_2,
            }
        except TypeError as e:
             # Log the type if stacking fails unexpectedly
             first_pv_type = type(valid_examples[0]["pixel_values"]) if valid_examples else 'N/A'
             logger.error(f"Collate fn: Stacking failed! Type Error: {e}. Type of pixel_values in first valid example: {first_pv_type}")
             # Re-raise or return empty dict
             return {}
        except Exception as e:
            logger.error(f"Collate fn: Unexpected error during stacking: {e}")
            return {}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers, # Use arg for num_workers
    )

    # Create validation dataloader if val_dataset exists
    val_dataloader = None
    if eval_dataset:
        logger.info("Creating validation dataloader...")
        val_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            shuffle=False, # No need to shuffle validation data
            collate_fn=collate_fn,
            batch_size=args.validation_batch_size, # Use specific validation batch size from args
            num_workers=args.dataloader_num_workers,
        )

    # --- Learning Rate Scheduler ---
    lr_scheduler = get_scheduler(
        "cosine", # Common scheduler type
        optimizer=optimizer,
        num_warmup_steps=50, # Example warmup steps
        num_training_steps=(len(train_dataloader) * args.epochs) // args.gradient_accumulation_steps,
    )

    # --- Prepare with Accelerator ---
    # Prepare relevant items (handle val_dataloader conditionally)
    logger.info("Preparing models and dataloaders with Accelerator...")
    if val_dataloader:
        transformer, optimizer, train_dataloader, lr_scheduler, val_dataloader = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler, val_dataloader
        )
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )

    # Move text_encoder to device
    text_encoder.to(accelerator.device)
    text_encoder_2.to(accelerator.device)

    # --- Calculate Training Steps --- 
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is not None and args.max_train_steps > 0:
        max_train_steps = args.max_train_steps
        args.epochs = math.ceil(max_train_steps / num_update_steps_per_epoch) # Recalculate epochs based on max_train_steps
        logger.info(f"Training for a fixed {max_train_steps} steps, overriding epochs to {args.epochs}.")
    else:
        max_train_steps = args.epochs * num_update_steps_per_epoch
        logger.info(f"Training for {args.epochs} epochs, corresponding to {max_train_steps} steps.")

    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num training examples = {len(train_dataset)}")
    if eval_dataset: # Check if eval_dataset was created before logging its length
        logger.info(f"  Num validation examples = {len(eval_dataset)}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0
    best_val_loss = float('inf') # Initialize best validation loss tracking

    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    # Record start time
    training_start_time = time.time()

    for epoch in range(first_epoch, args.epochs):
        transformer.train()
        train_loss = 0.0
        steps_in_epoch = 0 # Track optimization steps within the epoch
        images_processed_epoch = 0 # Track images processed in this epoch
        epoch_start_time = time.time() # Record time at the start of the epoch

        for step, batch in enumerate(train_dataloader):
            # Move pixel values to device/dtype
            pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
            batch_size = pixel_values.shape[0] # Get batch size from pixel_values

            # --- Prepare Conditional Inputs based on Dataset Type ---
            input_ids = None
            input_ids_2 = None
            prompt_embeds_2 = None
            clip_pooled = None
            if args.dataset_type == "imagefolder":
                # Image-only: Set text inputs to None
                logger.debug("Image-only batch detected. Using None for text inputs.")
            else:
                # Multimodal: Get text inputs from batch and encode
                input_ids_2_batch = batch["input_ids_2"].to(accelerator.device) if "input_ids_2" in batch and batch["input_ids_2"] is not None else None

                if input_ids_2_batch is not None:
                    logger.debug("Processing batch with T5 text inputs (input_ids_2).")
                    # Encode captions using T5 encoder (encoder_2)
                    prompt_embeds_2 = pipeline.text_encoder_2(input_ids_2_batch)[0]
                    # pooled_prompt_embeds is not directly used by FLUX transformer, skip generating it unless needed elsewhere
                else:
                    # This case should ideally not happen if all examples have captions and were tokenized
                    logger.warning("Batch found with missing 'input_ids_2'. Handling as image-only.")
                    prompt_embeds = None # Set to None if input_ids was removed
                    prompt_embeds_2 = None
                    pooled_projections = None
                    txt_ids = None # Set to None if input_ids was removed

                # Placeholder handling (Memory: 361714d3) - This logic needs review based on removing input_ids
                if prompt_embeds_2 is None:
                    # Need to know the expected cross_attention_dim for FLUX
                    cross_attn_dim = transformer.config.cross_attention_dim # Check this attribute exists
                    batch_size = pixel_values.shape[0]
                    prompt_embeds_2 = torch.zeros(
                        (batch_size, 1, cross_attn_dim),
                        dtype=weight_dtype, device=accelerator.device
                    )
                    logger.warning(f"Created null T5 embeds: {prompt_embeds_2.shape}")

                # Ensure input_ids_2 is None for image-only batches before the call
                if input_ids_2_batch is None:
                    # This confirmation might seem redundant if it comes in as None, 
                    # but ensures it's explicitly None before passing.
                    input_ids_2 = None 
                    logger.warning("Confirmed input_ids_2 is None for image-only batch.")

                # Create null T5 input IDs if still None
                if input_ids_2 is None:
                     input_ids_2 = torch.zeros(
                         (batch_size, 1),
                         dtype=torch.long, device=accelerator.device
                     )
                     logger.warning(f"input_ids_2 was None, created minimal placeholder: {input_ids_2.shape}")

            # --- Handle Missing Conditioning Inputs (Create Placeholders) --- #
            # Get expected embedding dimensions and null sequence length
            t5_embed_dim = getattr(transformer.config, 'cross_attention_dim', transformer.config.joint_attention_dim) # e.g., 4096
            null_sequence_length = 1 # Minimal length for null sequences

            # For non-imagefolder datasets, ensure text conditioning placeholders exist
            if args.dataset_type != "imagefolder":
                # Create null T5 embeddings if still None
                if prompt_embeds_2 is None:
                    prompt_embeds_2 = torch.zeros(
                        batch_size, null_sequence_length, t5_embed_dim,
                        dtype=weight_dtype, device=accelerator.device
                    )
                    logger.debug(f"Created null T5 embeds: {prompt_embeds_2.shape}")
                else:
                    prompt_embeds_2 = prompt_embeds_2.to(dtype=weight_dtype)

                # Create null CLIP pooled projections if still None
                if clip_pooled is None:
                    try:
                        # Try to get the specific dim expected by the text embedder for pooled projections
                        expected_clip_dim = transformer.config.pooled_projection_dim
                        if expected_clip_dim is None:
                            logger.warning("transformer.config.pooled_projection_dim is None, falling back to 768.")
                            expected_clip_dim = 768 # Fallback (common in some FLUX variants)
                    except AttributeError:
                        logger.warning("Could not find transformer.config.pooled_projection_dim, falling back to 768.")
                        expected_clip_dim = 768 # Fallback

                    clip_pooled = torch.zeros(batch_size, expected_clip_dim, dtype=weight_dtype, device=accelerator.device)
                    logger.warning(f"clip_pooled was None, created placeholder with expected dim {expected_clip_dim}: {clip_pooled.shape}")
                else:
                    clip_pooled = clip_pooled.to(dtype=weight_dtype)

            # --- VAE Encoding and Noise Addition (Common Logic) ---
            with accelerator.accumulate(transformer): # Use transformer here
                # Encode pixel values -> latents using VAE
                # VAE is prepared by accelerator, handles device/dtype internally via hooks/casting
                latents = vae.encode(pixel_values.to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                logger.debug(f"Initial VAE latents shape: {latents.shape}") # Log shape immediately after VAE

                latents = latents * vae.config.scaling_factor
                # No need for .to(accelerator.device) here, accelerator handles it

                # Apply projection layer if defined (Conv2d operates spatially)
                logger.debug(f"Shape before projection: {latents.shape}, Target input channels: {transformer_in_channels_actual}")
                if vae_to_transformer_projection is not None:
                    latents = vae_to_transformer_projection(latents) # Apply Conv2d: (B, C_in, H, W) -> (B, C_out, H, W)
                    logger.debug(f"Projected latents shape: {latents.shape}")

                # Reshape latents for transformer: (B, C, H, W) -> (B, H*W, C)
                bsz, channels, height, width = latents.shape # Use bsz here, batch_size might be different due to accumulation
                latents_reshaped = latents.permute(0, 2, 3, 1).reshape(bsz, height * width, channels)
                logger.debug(f"Shape AFTER reshape (Input to transformer) - latents_reshaped: {latents_reshaped.shape}")

                # Generate correct 1D img_ids (positional indices) and expand to batch size
                img_ids_1d = torch.arange(height * width, device=latents.device)
                img_ids = img_ids_1d.repeat(bsz, 1)
                logger.debug(f"Generated 1D img_ids shape: {img_ids.shape}")

                # Sample noise and timesteps
                noise = torch.randn_like(latents_reshaped)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents_reshaped.device)
                timesteps = timesteps.long()

                # Ensure pooled projections exist (create placeholder if needed) -- Moved here AGAIN to prevent logging error
                if clip_pooled is None:
                    # Get expected dimension from transformer config
                    try:
                        expected_clip_dim = transformer.config.pooled_projection_dim
                        if expected_clip_dim is None:
                            logger.warning("transformer.config.pooled_projection_dim is None, falling back to 768.")
                            expected_clip_dim = 768
                    except AttributeError:
                        logger.warning("Could not find transformer.config.pooled_projection_dim, falling back to 768.")
                        expected_clip_dim = 768
                    clip_pooled = torch.zeros(bsz, expected_clip_dim, dtype=weight_dtype, device=accelerator.device)
                    logger.warning(f"clip_pooled was None immediately before logging/transformer call, created placeholder with expected dim {expected_clip_dim}: {clip_pooled.shape}")

                # Ensure prompt_embeds_2 is a tensor, even for image-only datasets
                if prompt_embeds_2 is None:
                    cross_attention_dim = getattr(transformer.config, 'cross_attention_dim', None)
                    if cross_attention_dim is None:
                        cross_attention_dim = getattr(transformer.config, 'joint_attention_dim', None)
                    if cross_attention_dim is None:
                        raise ValueError("Could not determine cross_attention_dim for placeholder prompt_embeds_2.")
                    # Use seq_len=1 for null prompt
                    prompt_embeds_2 = torch.zeros(
                        bsz, 1, cross_attention_dim,
                        dtype=weight_dtype, device=accelerator.device
                    )
                    logger.warning(f"prompt_embeds_2 was None, created placeholder: {prompt_embeds_2.shape}")

                # Ensure input_ids_2 is None for image-only batches before the call
                if input_ids_2_batch is None:
                    # This confirmation might seem redundant if it comes in as None, 
                    # but ensures it's explicitly None before passing.
                    input_ids_2 = None 
                    logger.warning("Confirmed input_ids_2 is None for image-only batch.")

                # Create null T5 input IDs if still None
                if input_ids_2 is None:
                     input_ids_2 = torch.zeros(
                         bsz, 1,
                         dtype=torch.long, device=accelerator.device
                     )
                     logger.warning(f"input_ids_2 was None, created minimal placeholder: {input_ids_2.shape}")

                # Predict the noise residual using the transformer model
                # Pass the prepared conditional inputs (placeholders or None)
                logger.debug(f"  transformer input shape - hidden_states: {latents_reshaped.shape}")
                logger.debug(f"  transformer input shape - timestep: {timesteps.shape}") 
                logger.debug(f"  transformer input shape - encoder_hidden_states: {prompt_embeds_2.shape}") # Placeholder ensured
                logger.debug(f"  transformer input shape - pooled_projections: {clip_pooled.shape}")       # Placeholder ensured
                logger.debug(f"  transformer input shape - img_ids: {img_ids.shape}")
                # Log txt_ids shape only if it exists 
                if input_ids_2 is not None:
                     logger.debug(f"  transformer input shape - txt_ids: {input_ids_2.shape}")
                else:
                     logger.debug("  transformer input shape - txt_ids: None") # Explicitly logging None

                # Pass arguments explicitly
                model_pred = transformer(
                    hidden_states=latents_reshaped,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds_2,
                    pooled_projections=clip_pooled,
                    img_ids=img_ids,
                    txt_ids=input_ids_2 # Pass the placeholder tensor
                )

                # Assume prediction target is the noise (epsilon prediction)
                target = noise
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                # Accumulate loss for epoch averaging
                train_loss += avg_loss.item()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # Clip gradients if needed (helps prevent exploding gradients)
                    params_needing_grad = [p for p in params_to_optimize if p.grad is not None]
                    if params_needing_grad:
                         accelerator.clip_grad_norm_(params_needing_grad, 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                steps_in_epoch += 1 # Increment steps *after* gradient sync
                images_processed_epoch += total_batch_size # Accumulate total images processed
                accelerator.log({"train_loss_step": avg_loss.item()}, step=global_step)

                # Periodic checkpointing (optional, separate from best model saving)
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        # Save LoRA weights specifically if using PEFT library
                        unwrapped_transformer = accelerator.unwrap_model(transformer)
                        if isinstance(unwrapped_transformer, PeftModel):
                            # Use the save_pretrained method from the PeftModel instance
                            unwrapped_transformer.save_pretrained(save_path)
                        else:
                            # Fallback or add specific logic if using a custom model structure
                            torch.save(unwrapped_transformer.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
                        logger.info(f"Saved state and LoRA weights to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        # --- End of Training Steps for Epoch --- 

        # Wait for all processes to finish training steps before validation
        accelerator.wait_for_everyone()

        # --- Validation Phase --- 
        avg_val_loss = 0.0
        if eval_dataset:
            logger.info(f"Running validation for epoch {epoch}...")
            transformer.eval() # Use transformer
            val_loss = 0.0
            val_steps = 0
            for step, val_batch in enumerate(tqdm(val_dataloader, desc="Validation", disable=not accelerator.is_local_main_process)):
                with torch.no_grad():
                    # Prepare inputs for validation (similar to training)
                    # Ensure pixel_values are on the correct device and dtype
                    pixel_values_device = val_batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)

                    # Encode pixel values -> latents
                    # VAE is already on the correct device and dtype (float16)
                    latents = vae.encode(pixel_values_device).latent_dist.sample()
                    logger.debug(f"Validation: Initial VAE latents shape: {latents.shape}") # Log shape

                    latents = latents * vae.config.scaling_factor
                    latents = latents

                    # Apply projection layer if defined
                    logger.debug(f"Validation: Shape before projection: {latents.shape}, Target input channels: {transformer_in_channels_actual}")
                    if vae_to_transformer_projection is not None:
                        b, c, h, w = latents.shape
                        if c != vae_latent_channels_actual:
                            logger.error(f"PANIC: Validation Latent channels {c} != expected {vae_latent_channels_actual} before projection!")
                            raise ValueError(f"Unexpected validation latent channel dimension: {c}")
                            
                        latents_reshaped = latents.permute(0, 2, 3, 1).reshape(b * h * w, c) # Reshape for Linear layer (B*H*W, C_in=16)
                        projected_latents_reshaped = vae_to_transformer_projection(latents_reshaped) # Apply projection 16->64
                        latents = projected_latents_reshaped.reshape(b, h, w, transformer_in_channels_actual).permute(0, 3, 1, 2) # Reshape back (B, C_out=64, H, W)
                        logger.debug(f"Validation: Projected latents shape: {latents.shape}")
                        
                    # Reshape latents for transformer: (B, C, H, W) -> (B, H*W, C)
                    bsz_val, channels_val, height_val, width_val = latents.shape
                    latents_reshaped_val = latents.permute(0, 2, 3, 1).reshape(bsz_val, height_val * width_val, channels_val)
                    logger.debug(f"Validation shape after reshape: {latents_reshaped_val.shape}")

                    # Handle case where 'caption' is not in the batch (image-only dataset)
                    input_ids_2 = None # Initialize as None
                    prompt_embeds_2 = None
                    clip_pooled = None
                    if 'input_ids_2' not in val_batch or val_batch['input_ids_2'] is None:
                        logger.debug("Validation: Image-only batch. Using None for text inputs.")
                        # Text inputs remain None
                    else:
                        logger.debug("Validation: Multimodal batch. Encoding text.")
                        input_ids_2 = val_batch['input_ids_2'].to(accelerator.device)

                    # Encode prompts for validation batch
                    with torch.no_grad():
                        # CLIP Embeddings (Assume 'input_ids' always exists for CLIP pooled)
                        clip_input_ids = val_batch.get("input_ids")
                        if clip_input_ids is not None:
                            prompt_embeds_outputs = text_encoder(clip_input_ids.to(accelerator.device), output_hidden_states=True)
                            clip_pooled = prompt_embeds_outputs.pooler_output
                        else: # Should not happen if dataset has 'text' col, but handle defensively
                            logger.warning("Validation: Missing 'input_ids' for CLIP pooled calculation!")
                            clip_pooled = None # Ensure it's None if input is missing

                        # T5 Embeddings
                        if input_ids_2 is not None:
                            prompt_embeds_2_outputs = text_encoder_2(input_ids_2, output_hidden_states=True)
                            prompt_embeds_2 = prompt_embeds_2_outputs.last_hidden_state
                        # Else: prompt_embeds_2 remains None

                    # Generate correct 1D img_ids for validation
                    seq_len_val = height_val * width_val
                    img_ids_1d_val = torch.arange(seq_len_val, device=latents.device)
                    img_ids_val = img_ids_1d_val.repeat(bsz_val, 1) # Use bsz_val
                    logger.debug(f"Generated validation 1D img_ids shape: {img_ids_val.shape}")

                    # Sample noise and timesteps for validation
                    noise = torch.randn_like(latents_reshaped_val)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Predict noise using the model
                    model_pred_val = transformer(
                        hidden_states=latents_reshaped_val,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds_2, # T5 sequence embeds (None for img-only)
                        pooled_projections=clip_pooled, # CLIP pooled embeds (placeholder for img-only)
                        txt_ids=input_ids_2, # Pass the variable prepared earlier (None for image-only)
                        img_ids=img_ids_val, # Use corrected 1D validation img_ids
                    ).sample

                    # Assume target is noise for validation loss calculation
                    target = noise
                    loss = F.mse_loss(model_pred_val.float(), target.float(), reduction="mean")

                    val_loss += loss.item()
                    val_steps += 1

            # Calculate average validation loss for the epoch
            avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0

            # Check if this is the best validation loss so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if accelerator.is_main_process:
                    # Save the best model checkpoint (LoRA weights or full model)
                    best_checkpoint_dir = os.path.join(args.output_dir, "best_checkpoint")
                    os.makedirs(best_checkpoint_dir, exist_ok=True)
                    logger.info(f"Saving best model checkpoint to {best_checkpoint_dir} (Val Loss: {best_val_loss:.4f})...")
                    
                    unwrapped_transformer = accelerator.unwrap_model(transformer)
                    if isinstance(unwrapped_transformer, PeftModel):
                        # Save using appropriate method (PEFT LoRA or standard HF)
                        unwrapped_transformer.save_pretrained(best_checkpoint_dir)
                    elif hasattr(unwrapped_transformer, 'save_pretrained'):
                        # Standard Hugging Face save_pretrained for non-PEFT models or if LoRA is merged
                        unwrapped_transformer.save_pretrained(best_checkpoint_dir)
                    else:
                        # Fallback or add specific logic if using a custom model structure
                        torch.save(unwrapped_transformer.state_dict(), os.path.join(best_checkpoint_dir, "pytorch_model.bin"))
                    logger.info(f"New best validation loss: {best_val_loss:.4f}. Saved checkpoint to {best_checkpoint_dir}")
            # Set model back to train mode after validation
            transformer.train()
        # --- End of Validation Phase --- 

        # Calculate average training loss and throughput for the epoch
        avg_train_loss_epoch = train_loss / steps_in_epoch if steps_in_epoch > 0 else 0.0
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time # Correctly use epoch start/end times
        train_throughput = images_processed_epoch / epoch_duration if epoch_duration > 0 else 0

        # Log epoch metrics
        if accelerator.is_main_process:
            log_metrics = {
                "epoch": epoch,
                "train_loss_epoch": avg_train_loss_epoch,
                "train_throughput_img_sec": train_throughput,
            }
            if eval_dataset:
                log_metrics["val_loss_epoch"] = avg_val_loss
            
            logger.info(f"Epoch {epoch} Summary: {log_metrics}")
            # Log metrics to configured tracker (e.g., wandb)
            accelerator.log(log_metrics, step=global_step) # Log with global_step for finer granularity if needed

        if global_step >= max_train_steps:
            logger.info("Reached max_train_steps. Stopping training.")
            break

    # Record end time and calculate metrics
    training_end_time = time.time()
    total_training_time_seconds = training_end_time - training_start_time
    final_global_step = global_step # Capture final step count
    avg_time_per_step_seconds = total_training_time_seconds / final_global_step if final_global_step > 0 else 0
    num_training_images = len(train_dataset)

    accelerator.wait_for_everyone()

    # Log training summary metrics
    if accelerator.is_main_process:
        logger.info("***** Final Training Summary *****")
        logger.info(f"  Total training steps: {final_global_step}")
        logger.info(f"  Total training time: {total_training_time_seconds:.2f} seconds ({total_training_time_seconds/3600:.2f} hours)")
        logger.info(f"  Average time per step: {avg_time_per_step_seconds:.4f} seconds")
        logger.info(f"  Number of training images used: {num_training_images}")
        if eval_dataset:
            logger.info(f"  Best validation loss achieved: {best_val_loss:.4f}")
        logger.info(f"  Number of epochs completed: {epoch + 1 if 'epoch' in locals() else 'N/A'} / {args.epochs}")

    # --- Save the Final Model ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_model = accelerator.unwrap_model(transformer)
        if isinstance(final_model, PeftModel):
            final_model.save_pretrained(args.output_dir)
        else:
            logger.warning("Final model is not a PeftModel, attempting standard save_pretrained.")
            if hasattr(final_model, 'save_pretrained'):
                 final_model.save_pretrained(args.output_dir)
            else:
                # Fallback or add specific logic if using a custom model structure
                torch.save(final_model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
        logger.info(f"Saved final LoRA weights (or full model) to {args.output_dir}")

    accelerator.end_training()
    end_time = time.time()
    total_duration = end_time - training_start_time
    logger.info(f"Training finished in {total_duration:.2f} seconds.")

def log_transformer_inputs(logger, hidden_states, timestep, encoder_hidden_states, pooled_projections, txt_ids, img_ids, prefix="Training"):
    logger.debug(f"--- {prefix} Transformer Inputs ---")
    logger.debug(f"  hidden_states shape: {hidden_states.shape}")
    logger.debug(f"  timestep shape: {timestep.shape}")
    logger.debug(f"  pooled_projections shape: {pooled_projections.shape}")
    logger.debug(f"  txt_ids type: {type(txt_ids)}")
    logger.debug(f"  txt_ids shape: {txt_ids.shape if txt_ids is not None else 'None'}")
    logger.debug(f"  img_ids shape: {img_ids.shape}")

if __name__ == "__main__":
    # Configuration loading and argument parsing are now handled within parse_args
    args = parse_args()
    
    # Defaults were already handled inside parse_args
    # args.validation_batch_size = getattr(args, 'validation_batch_size', args.batch_size)
    # args.dataloader_num_workers = getattr(args, 'dataloader_num_workers', 4)
    # args.preprocessing_num_workers = getattr(args, 'preprocessing_num_workers', 1)

    main(args)
