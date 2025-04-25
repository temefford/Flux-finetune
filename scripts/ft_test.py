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
import logging
import traceback

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
def preprocess_single_example(example, dataset_abs_path, image_transforms, image_column, caption_column, hash_column, tokenizer_2):
    """Preprocesses a single example for training."""
    logger = logging.getLogger("preprocess_single")
    # logger.setLevel(logging.DEBUG) # Uncomment for verbose preprocessing logs

    image_hash = None
    image_path = None
    caption = ""

    try:
        # --- Determine Image Path --- #
        if hash_column and hash_column in example and isinstance(example[hash_column], str) and example[hash_column]:
            image_hash = example[hash_column]
            image_path = os.path.join(dataset_abs_path, f"{image_hash}.jpg")
        elif image_column in example:
            # Handle ImageFolder case where example[image_column] is a PIL Image object
            if isinstance(example[image_column], Image.Image):
                image = example[image_column].convert("RGB")
                # Use a placeholder name if hash isn't available for logging
                image_name_for_log = f"image_from_{image_column}"
            # Handle case where image_column contains a relative path/filename
            elif isinstance(example[image_column], str):
                filename = example[image_column]
                # Construct path relative to dataset_abs_path
                image_path = os.path.join(dataset_abs_path, filename if '.' in filename else f"{filename}.jpg")
                image_name_for_log = os.path.basename(image_path)
            else:
                logger.warning(f"Unexpected type in image_column: {type(example[image_column])}. Skipping example.")
                return {"pixel_values": None, "input_ids_2": None}
        else:
            logger.warning(f"Missing required image identifier ('{image_column}' or '{hash_column}'). Skipping example: {example}")
            return {"pixel_values": None, "input_ids_2": None}

        # --- Load Image if path was determined --- #
        if image_path:
            logger.debug(f"Loading image from path: {image_path}")
            try:
                image = Image.open(image_path).convert("RGB")
                image_name_for_log = os.path.basename(image_path)
            except FileNotFoundError:
                logger.warning(f"Image file not found: {image_path}. Skipping example.")
                return {"pixel_values": None, "input_ids_2": None}
            except Exception as img_load_err:
                logger.warning(f"Error loading image {image_path}: {img_load_err}. Skipping example.")
                return {"pixel_values": None, "input_ids_2": None}
        elif not 'image' in locals(): # Check if image was loaded via ImageFolder case
             logger.error("Image object not loaded and image_path is None. This shouldn't happen. Skipping example.")
             return {"pixel_values": None, "input_ids_2": None}

        # --- Image Transformation --- #
        pixel_values_tensor = None
        try:
            logger.debug(f"Applying image transforms to {image_name_for_log}: Mode={image.mode}, Size={image.size}, Format={getattr(image, 'format', 'N/A')}")
            # Pass the single PIL image to the transform function
            transformed_output = image_transforms(image) # Expects PIL, returns dict or tensor

            if isinstance(transformed_output, torch.Tensor):
                pixel_values_tensor = transformed_output
            elif isinstance(transformed_output, dict) and 'pixel_values' in transformed_output:
                pv_maybe_numpy_or_tensor = transformed_output['pixel_values']
                # Need to handle potential batch dim if transform adds one
                if isinstance(pv_maybe_numpy_or_tensor, np.ndarray):
                    pv_maybe_numpy_or_tensor = torch.from_numpy(pv_maybe_numpy_or_tensor)

                if isinstance(pv_maybe_numpy_or_tensor, torch.Tensor):
                    if pv_maybe_numpy_or_tensor.ndim == 4 and pv_maybe_numpy_or_tensor.shape[0] == 1:
                        pixel_values_tensor = pv_maybe_numpy_or_tensor.squeeze(0)
                    elif pv_maybe_numpy_or_tensor.ndim == 3:
                        pixel_values_tensor = pv_maybe_numpy_or_tensor
                    else:
                        logger.warning(f"Unexpected tensor shape from image transform: {pv_maybe_numpy_or_tensor.shape}. Skipping example.")
                        return {"pixel_values": None, "input_ids_2": None}
                else:
                     logger.warning(f"Unexpected type for pixel_values in dict: {type(pv_maybe_numpy_or_tensor)}. Skipping example.")
                     return {"pixel_values": None, "input_ids_2": None}
            else:
                 logger.warning(f"Unexpected output type from image_transforms: {type(transformed_output)}. Skipping example.")
                 return {"pixel_values": None, "input_ids_2": None}

            logger.debug(f"Successfully transformed image {image_name_for_log}. Shape: {pixel_values_tensor.shape}, Dtype: {pixel_values_tensor.dtype}")

        except Exception as img_proc_err:
            logger.warning(f"Error applying image transforms to {image_name_for_log}: {img_proc_err}. Skipping example.", exc_info=True)
            return {"pixel_values": None, "input_ids_2": None}

        # --- Text Tokenization --- #
        if caption_column and caption_column in example:
            caption = str(example[caption_column]) if example[caption_column] is not None else ""
        # For ImageFolder, caption might be missing, default to empty string
        else:
             caption = ""

        try:
            max_len = getattr(tokenizer_2, 'model_max_length', 512)
            text_inputs = tokenizer_2(
                caption, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt"
            )
            input_ids_2_tensor = text_inputs['input_ids'].squeeze(0) # Remove batch dimension
            logger.debug(f"Successfully tokenized caption for {image_name_for_log}. Shape: {input_ids_2_tensor.shape}")

        except Exception as txt_proc_err:
            logger.warning(f"Error tokenizing caption for {image_name_for_log}: '{caption[:50]}...'. Error: {txt_proc_err}. Skipping example.")
            return {"pixel_values": None, "input_ids_2": None}

        # --- Return Processed Example --- #
        result = {}
        if pixel_values_tensor is not None:
            result["pixel_values"] = pixel_values_tensor
            logger.debug(f"Preprocess check - pixel_values type: {type(result['pixel_values'])}, shape: {result['pixel_values'].shape if isinstance(result['pixel_values'], torch.Tensor) else 'N/A'}")
        else:
            logger.debug("Preprocess check - pixel_values is None")

        if input_ids_2_tensor is not None:
            result["input_ids_2"] = input_ids_2_tensor
            logger.debug(f"Preprocess check - input_ids_2 type: {type(result['input_ids_2'])}, shape: {result['input_ids_2'].shape if isinstance(result['input_ids_2'], torch.Tensor) else 'N/A'}")
        else:
            logger.debug("Preprocess check - input_ids_2 is None")

        return result

    except Exception as e:
        logger.error(f"General error processing example: {example}. Error: {e}", exc_info=True)
        return {"pixel_values": None, "input_ids_2": None}

# --- Main Function ---
def main(args):
    # === Configure Logging ===
    # Set level to DEBUG to capture image property logs
    log_level = logging.DEBUG
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(level=log_level, format=log_format)
    # ========================

    # Initialize Accelerator
    logging_dir = Path(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard", # or "wandb" if configured
        project_config=accelerator_project_config,
    )

    # === Log Logging Level (After Accelerator Init) ===
    logger.info(f"Logging level set to {logging.getLevelName(log_level)}")
    # ===================================================

    accelerator.print(f"DEBUG: Effective args.data_dir after parsing: {args.data_dir}")

    # Make one log on every process with the configuration for debugging.
    # logger.info(accelerator.state, main_process_only=False)

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
        _preprocess_func = partial(
            preprocess_single_example, # Use the NEW single-example function
            dataset_abs_path=args.dataset_path,
            # Pass the image processor's preprocess method directly
            image_transforms=pipeline.image_processor.preprocess,
            image_column=args.image_column,
            hash_column=getattr(args, 'hash_column', None), # Pass hash_column if defined, else None
            caption_column=args.caption_column,
            tokenizer_2=tokenizer_2,
        )

        logger.info(f"Preprocessing dataset using {args.preprocessing_num_workers} workers...")
        # Map the single-example function, NOT batched
        processed_dataset = dataset.map(
            _preprocess_func,
            batched=False, # Process one example at a time
            num_proc=args.preprocessing_num_workers,
            remove_columns=columns_to_remove,
            desc="Preprocessing dataset",
        )
        logger.info("Dataset preprocessing complete.")

    elif args.dataset_type == "imagefolder":
        # Assumes imagefolder dataset structure
        logger.info(f"Processing imagefolder dataset: {args.dataset_name}")
        original_columns = list(dataset['train'].features.keys())
        # Keep only the columns generated by preprocessing
        columns_to_keep = ['pixel_values', 'input_ids_2']
        columns_to_remove = [col for col in original_columns if col not in columns_to_keep]
        logger.info(f"Columns to remove: {columns_to_remove}")

        # Use the same single-example preprocessor
        _preprocess_func = partial(
             preprocess_single_example, # Use the NEW single-example function
             dataset_abs_path=args.dataset_path, # ImageFolder needs the root path
             image_transforms=pipeline.image_processor.preprocess,
             image_column=args.image_column,
             caption_column=None, # No captions expected
             hash_column=None,    # No hash expected
             tokenizer_2=tokenizer_2, # Still needed for dummy text IDs
        )
        processed_dataset = dataset.map(
             _preprocess_func,
             batched=False, # Process one example at a time
             num_proc=args.preprocessing_num_workers,
             remove_columns=columns_to_remove,
             desc="Preprocessing imagefolder dataset",
        )
        logger.info("Imagefolder preprocessing complete.")
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
        """Collates preprocessed examples into batches, filtering out invalid entries."""
        logger = logging.getLogger("collate_fn")
        # logger.setLevel(logging.DEBUG) # Optional: Set specific level for this logger

        # Filter out entries where pixel_values is None (indicating a preprocessing failure for that example)
        # Now examples should be dicts with Tensors or Nones
        valid_examples = [ex for ex in examples if ex is not None and ex.get("pixel_values") is not None and ex.get("input_ids_2") is not None]

        if not valid_examples:
            logger.warning("Collate function received batch with no valid examples after filtering Nones. Skipping batch.")
            # Return an empty dictionary or None to signal skipping this batch in the training loop
            return None # Return None to signal skipping

        # Log details of valid examples before stacking
        logger.debug(f"Collate - Processing {len(valid_examples)} valid examples out of {len(examples)} original.")
        # for i, example in enumerate(valid_examples):
        #     pv = example.get("pixel_values")
        #     ids2 = example.get("input_ids_2")
        #     pv_type = type(pv).__name__
        #     ids2_type = type(ids2).__name__
        #     # Use getattr to safely get shape, defaulting to 'N/A' if not a tensor
        #     pv_shape = getattr(pv, 'shape', 'N/A')
        #     ids2_shape = getattr(ids2, 'shape', 'N/A')
        #     logger.debug(f"Collate - Valid Example {i}: pixel_values type={pv_type}, shape={pv_shape}; input_ids_2 type={ids2_type}, shape={ids2_shape}")

        try:
            # Examples in valid_examples should now have Tensors directly
            pixel_values = torch.stack([example["pixel_values"] for example in valid_examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

            input_ids_2 = torch.stack([example["input_ids_2"] for example in valid_examples])

            # Return the batch dictionary for the model
            batch = {
                "pixel_values": pixel_values,
                "input_ids_2": input_ids_2,
            }
            return batch
        except TypeError as e:
            logger.error(f"Error during collate_fn stacking: {e}")
            # ENHANCED LOG: Log the types of the first few problematic items in the list before stacking
            logger.error(f"  Collate Troubleshoot - Attempting to stack {len([example['pixel_values'] for example in valid_examples])} pixel_values items.")
            for i, pv_item in enumerate([example['pixel_values'] for example in valid_examples][:5]): # Log first 5 pixel_values items
                logger.error(f"    Item {i} in pixel_values_list type: {type(pv_item)}")
            logger.error(f"  Collate Troubleshoot - Attempting to stack {len([example['input_ids_2'] for example in valid_examples])} input_ids_2 items.")
            for i, iid2_item in enumerate([example['input_ids_2'] for example in valid_examples][:5]): # Log first 5 input_ids_2 items
                 logger.error(f"    Item {i} in input_ids_2_list type: {type(iid2_item)}")

            return None # Return None if stacking fails
        except Exception as e:
            logger.error(f"Unexpected error in collate_fn: {e}", exc_info=True)
            # Log tensor shapes for debugging if possible
            for i, ex in enumerate(valid_examples):
                pv_shape = ex['pixel_values'].shape if isinstance(ex.get('pixel_values'), torch.Tensor) else 'Not Tensor or None'
                id_shape = ex['input_ids_2'].shape if isinstance(ex.get('input_ids_2'), torch.Tensor) else 'Not Tensor or None'
                logger.error(f"  Example {i} shapes - pixel_values: {pv_shape}, input_ids_2: {id_shape}")
            return None # Skip batch if stacking fails

    logger.info("Creating train DataLoader...")
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
            # === Check for skipped batch first ===
            if batch is None:
                logger.warning(f"Skipping batch {step} in epoch {epoch} due to previous collation error.")
                continue

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
                        # Try to get the specific dim expected by the text embedder for pooled projections
                        expected_clip_dim = transformer.config.pooled_projection_dim
                        if expected_clip_dim is None:
                            logger.warning("transformer.config.pooled_projection_dim is None, falling back to 768.")
                            expected_clip_dim = 768 # Fallback (common in some FLUX variants)
                    except AttributeError:
                        logger.warning("Could not find transformer.config.pooled_projection_dim, falling back to 768.")
                        expected_clip_dim = 768 # Fallback

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
            vae.eval()
            text_encoder_2.eval() # Set once before the loop
            total_val_loss = 0.0 # Initialize total validation loss
            val_steps = 0

            with torch.no_grad(): # Ensure no gradients are computed
                for val_step, val_batch in enumerate(val_dataloader):
                    # === Check for skipped validation batch first ===
                    if val_batch is None:
                        logger.warning(f"Skipping validation batch {val_step} due to collation error.")
                        continue

                    pixel_values_device = val_batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                    bsz = pixel_values_device.shape[0]

                    # Prepare validation inputs (similar logic to training)
                    input_ids_2_device = None
                    prompt_embeds_2_device = None
                    clip_pooled_device = None
                    if 'input_ids_2' not in val_batch or val_batch['input_ids_2'] is None:
                        logger.debug("Validation: Image-only batch. Using None for text inputs.")
                        # Text inputs remain None
                    else:
                        logger.debug("Validation: Multimodal batch. Encoding text.")
                        input_ids_2_device = val_batch['input_ids_2'].to(accelerator.device)

                    # Encode prompts for validation batch
                    with torch.no_grad():
                        # CLIP Embeddings (Assume 'input_ids' always exists for CLIP pooled)
                        clip_input_ids = val_batch.get("input_ids")
                        if clip_input_ids is not None:
                            prompt_embeds_outputs = text_encoder(clip_input_ids.to(accelerator.device), output_hidden_states=True)
                            clip_pooled_device = prompt_embeds_outputs.pooler_output
                        else: # Should not happen if dataset has 'text' col, but handle defensively
                            logger.warning("Validation: Missing 'input_ids' for CLIP pooled calculation!")
                            clip_pooled_device = None # Ensure it's None if input is missing

                        # T5 Embeddings
                        if input_ids_2_device is not None:
                            prompt_embeds_2_outputs = text_encoder_2(input_ids_2_device, output_hidden_states=True)
                            prompt_embeds_2_device = prompt_embeds_2_outputs.last_hidden_state
                        # Else: prompt_embeds_2 remains None

                    # Generate correct 1D img_ids for validation
                    seq_len_val = bsz * 64 # Assuming 64x64 images
                    img_ids_1d_val = torch.arange(seq_len_val, device=latents.device)
                    img_ids_val = img_ids_1d_val.repeat(bsz, 1) # Use bsz
                    logger.debug(f"Generated validation 1D img_ids shape: {img_ids_val.shape}")

                    # Sample noise and timesteps for validation
                    noise = torch.randn_like(latents_reshaped)
                    bsz = pixel_values_device.shape[0]
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Predict noise using the model
                    model_pred_val = transformer(
                        hidden_states=latents_reshaped,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds_2_device, # T5 sequence embeds (None for img-only)
                        pooled_projections=clip_pooled_device, # CLIP pooled embeds (placeholder for img-only)
                        txt_ids=input_ids_2_device, # Pass the variable prepared earlier (None for image-only)
                        img_ids=img_ids_val, # Use corrected 1D validation img_ids
                    ).sample

                    # Assume target is noise for validation loss calculation
                    target = noise
                    loss = F.mse_loss(model_pred_val.float(), target.float(), reduction="mean")

                    val_loss = loss.item() # Get loss for the current batch
                    total_val_loss += val_loss # Accumulate loss
                    val_steps += 1

            # Calculate average validation loss after the loop
            if val_steps > 0:
                avg_val_loss = total_val_loss / val_steps
                logger.info(f"Validation Loss: {avg_val_loss:.4f} after {val_steps} steps.")
                # Track best validation loss
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