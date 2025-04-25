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



###############################################################################
# helper: build txt-position ids that have the same width as img_ids
###############################################################################
def make_txt_ids(img_ids: torch.Tensor) -> torch.LongTensor:
    """
    Given `img_ids` of shape (B , W) return a LongTensor of the same shape
    that simply contains 0..W-1 repeated for each batch element.
    """
    seq_len = img_ids.shape[1]
    base    = torch.arange(seq_len, device=img_ids.device)
    return base.unsqueeze(0).repeat(img_ids.shape[0], 1)  # (B , W)


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
    parser.add_argument("--val_split", type=float, default=None, help="Validation split ratio (overrides config).")
    parser.add_argument("--validation_split", type=float, default=None, help="(Legacy) Validation split ratio (overrides config).")
    parser.add_argument("--log_level", type=str, default=None, help="Logging level (overrides config).") # Added for consistency
    parser.add_argument(
        "--caption_column", type=str, default="text", help="The name of the caption column in the dataset."
    )
    parser.add_argument(
        "--hash_column", type=str, default="hash", help="The name of the hash column in the dataset (used as dummy text input)."
    )
    cmd_args = parser.parse_args() # Parse command line args first

    # --- Load Configuration from YAML --- #
    # --- Handle conflicting split flags ---
    if cmd_args.val_split is not None and cmd_args.validation_split is not None:
        raise ValueError("Specify only one of val_split or validation_split")
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
    elif cmd_args.val_split is not None: 
        args.validation_split = cmd_args.val_split
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
def preprocess_train(examples, dataset_abs_path, image_transforms, image_column, caption_column, hash_column, tokenizer_2):
    logger.info(f"[preprocess_train] dataset_abs_path: {dataset_abs_path}")
    """Preprocesses a batch of examples for training."""
    # Determine image paths and handle potential hash column presence
    # Image files are expected directly in dataset_abs_path, alongside metadata
    # Build a *correct* image root once and reuse it
     # <-- ADAPT if your folder is different
    # 1️⃣  make sure hash_column is usable
    if not hash_column:
        hash_column = image_column        # <- use image_column as backup
    image_root = dataset_abs_path
    if hash_column and hash_column in examples:
        image_paths = [os.path.join(image_root, f"{fn}.jpg") for fn in examples[image_column]]
        logger.info(f"[preprocess_train] Example image paths: {image_paths[:5]}")
    elif image_column in examples:
        # Assuming image_column contains filenames like 'image_001.jpg'
        # Construct path using dataset_abs_path directly
        image_paths = [os.path.join(image_root, f"{fn}.jpg") for fn in examples[image_column]]
        logger.info(f"[preprocess_train] Example image paths: {image_paths[:5]}")
    else:
        logger.error(f"Missing required image identifier column ('{image_column}' or '{hash_column}') in examples.")
        return {"pixel_values": [None] * len(examples.get(list(examples.keys())[0], [])), "input_ids_2": [None] * len(examples.get(list(examples.keys())[0], []))}

    original_batch_size = len(image_paths)
    # Initialize output lists with Nones
    pixel_values_list = [None] * original_batch_size
    input_ids_list = [None] * original_batch_size

    try:
        # --- Load Images Individually with Error Handling --- #
        images = [None] * original_batch_size # Pre-allocate list for images or Nones
        valid_indices = [] # Indices of successfully loaded images
        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
                images[i] = img
                valid_indices.append(i)
            except IndexError as ie:
                logger.warning(f"IndexError loading/converting image: {path}. Error: {ie}. Skipping.")
            except FileNotFoundError:
                 logger.warning(f"Image file not found during preprocessing: {path}. Skipping.")
            except Exception as img_err:
                 logger.warning(f"Error loading/converting image {path}: {img_err}. Skipping.")

        # Filter out None entries to get list of valid PIL images
        valid_images = [images[i] for i in valid_indices]

        if not valid_images:
            logger.warning("No valid images could be loaded in this batch.")
            # Return lists of Nones matching original batch size
            return {"pixel_values": pixel_values_list, "input_ids_2": input_ids_list}

        # --- Image Processing --- #

        pixel_values_valid_tensor = None
        try:
            # Attempt batch processing first
            image_inputs = image_transforms(valid_images)
            logger.debug(f"Type of image_inputs: {type(image_inputs)}")
            if isinstance(image_inputs, dict) and 'pixel_values' in image_inputs:
                pixel_values_maybe_numpy = image_inputs['pixel_values']
                # <- fall through to conversion logic below
            elif isinstance(image_inputs, torch.Tensor):
                # New – current diffusers returns a tensor directly
                pixel_values_valid_tensor = image_inputs          # [B,C,H,W]
            elif isinstance(image_inputs, np.ndarray):
                pixel_values_valid_tensor = torch.from_numpy(image_inputs)
            else:
                logger.warning(
                    f"image_inputs of unexpected type: {type(image_inputs)}  "
                    "(neither dict nor Tensor); skipping batch."
                )
                raise TypeError("Unexpected image_inputs type")

        except IndexError as batch_ie:
            logger.warning(f"Batch image processing failed with IndexError: {batch_ie}. Falling back to individual processing to identify culprit(s).")
            # Fallback: Process images individually to find the problematic one
            processed_individual_tensors = {}
            for i, img_idx in enumerate(valid_indices):
                img_to_process = images[img_idx]
                img_path = image_paths[img_idx]
                try:
                    individual_input = image_transforms([img_to_process]) # Process as a batch of 1

                    # Handle direct tensor output or dictionary output
                    pv_individual = None
                    if isinstance(individual_input, torch.Tensor):
                        # If image_transforms returns a tensor directly
                        pv_individual = individual_input.squeeze(0) # Assume it might still have a batch dim of 1
                    elif isinstance(individual_input, dict) and 'pixel_values' in individual_input:
                        # If image_transforms returns a dict (original expectation)
                        pv_maybe_numpy_or_tensor = individual_input['pixel_values']
                        if isinstance(pv_maybe_numpy_or_tensor, np.ndarray):
                            pv_individual = torch.from_numpy(pv_maybe_numpy_or_tensor).squeeze(0) # Remove batch dim
                        elif isinstance(pv_maybe_numpy_or_tensor, torch.Tensor):
                             pv_individual = pv_maybe_numpy_or_tensor.squeeze(0) # Remove batch dim
                        else:
                            logger.warning(f"Fallback - Unexpected type for pixel_values in dict: {type(pv_maybe_numpy_or_tensor)}")
                        if pv_individual is not None:
                            pass
                        else:
                            logger.error(f"Fallback - Unexpected output type from image_transforms: {type(individual_input)}")

                    if pv_individual is not None:
                        processed_individual_tensors[img_idx] = pv_individual
                    else:
                         logger.warning(f"Fallback - Failed to extract tensor for image {os.path.basename(img_path)}")


                except Exception as individual_e: # Catch specific errors if needed
                    # Use traceback for more detailed error logging in fallback
                    tb_str = traceback.format_exc()
                    logger.error(f"--> Error <-- Fallback processing failed for image {img_path}. Error: {individual_e}\nTraceback:\n{tb_str}. Skipping.")
                    # processed_individual_tensors[img_idx] remains None

            # Reconstruct the batch tensor from successfully processed individual images
            valid_tensors_list = [processed_individual_tensors.get(idx) for idx in valid_indices if processed_individual_tensors.get(idx) is not None]
            if not valid_tensors_list:
                 logger.error("No images could be processed individually after batch failure.")
                 return {"pixel_values": pixel_values_list, "input_ids_2": input_ids_list}
            pixel_values_valid_tensor = torch.stack(valid_tensors_list)
            # Update valid_indices to only include successfully processed ones *after fallback*
            valid_indices = [idx for idx in valid_indices if processed_individual_tensors.get(idx) is not None]
            if not valid_indices:
                logger.error("Valid indices list is empty after individual processing fallback.")
                return {"pixel_values": pixel_values_list, "input_ids_2": input_ids_list}

        except Exception as process_e:
            logger.error(f"General error during image processing: {process_e}")
            return {"pixel_values": pixel_values_list, "input_ids_2": input_ids_list}

        # If pixel_values_valid_tensor is still None here, something went wrong
        if pixel_values_valid_tensor is None:
             logger.error("pixel_values_valid_tensor is unexpectedly None after image processing block.")
             return {"pixel_values": pixel_values_list, "input_ids_2": input_ids_list}

        # --- Text Processing (Batched) ---
        # Ensure we use the potentially updated valid_indices from fallback processing
        valid_captions = [str(examples[caption_column][i]) if examples[caption_column][i] is not None else "" for i in valid_indices]

        # Process only the valid captions
        max_len = getattr(tokenizer_2, 'model_max_length', 512)
        if not valid_captions:
             # This case should ideally not happen if valid_images exist, but check anyway
             logger.error("No valid captions found corresponding to valid images.")
             return {"pixel_values": pixel_values_list, "input_ids_2": input_ids_list}

        text_inputs = tokenizer_2(
            valid_captions, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt"
        )
        input_ids_valid_tensor = text_inputs['input_ids'] # Tensor [NumValid, SeqLen]

        # --- Sanity Check Batch Sizes (Should match num valid images) ---
        num_valid = len(valid_images)
        if pixel_values_valid_tensor.shape[0] != num_valid or input_ids_valid_tensor.shape[0] != num_valid:
            logger.error(f"Batch size mismatch after processing valid items in preprocess_train. Expected {num_valid}, got {pixel_values_valid_tensor.shape[0]} images and {input_ids_valid_tensor.shape[0]} texts. Skipping batch.")
            return {"pixel_values": pixel_values_list, "input_ids_2": input_ids_list}

        # --- Distribute Valid Tensors into Full-Sized Lists ---
        valid_item_idx = 0
        for original_idx in valid_indices:
            # Detach tensors before putting them in the list if they require gradients (unlikely here, but good practice)
            # AFTER   (no .tolist(), stays a Tensor)
            # AFTER   ─ convert to NumPy (or .tolist()) so HF-Datasets can store it
            pixel_values_list[original_idx] = (
                    pixel_values_valid_tensor[valid_item_idx].cpu().numpy()
            )
            input_ids_list[original_idx]    = (
                    input_ids_valid_tensor[valid_item_idx].cpu().numpy()
            )
            valid_item_idx += 1
        # --- Return Lists for Dataset Map --- #
        return {                   # HF datasets will happily keep lists
            "pixel_values": pixel_values_list,
            "input_ids_2": input_ids_list,
        }

    except Exception as e:
        logger.error(f"General error during preprocessing batch: {e}", exc_info=True)
        # Return lists of Nones on general failure
        return {"pixel_values": [None] * original_batch_size, "input_ids_2": [None] * original_batch_size}

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
    # === Configure Logging ===
    # Set level to DEBUG to capture image property logs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    level = getattr(logging, getattr(args, "log_level", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, format=log_format)

    # Initialize Accelerator
    logging_dir = Path(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard", # or "wandb" if configured
        project_config=accelerator_project_config,
    )

    # Now initialize logger after Accelerator (accelerate state) is ready
    global logger
    logger = get_logger(__name__)
    logger.info(f"Logging level set to {logging.getLevelName(level)}")
    # ========================

    accelerator.print(f"DEBUG: Effective args.dataset_path after parsing: {args.dataset_path}")

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
        target_modules = ["to_q","to_k","to_v"]
        transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank, # Often set equal to rank
            # Use correct FLUX attention module names (to_q, to_k, to_v)
            target_modules=target_modules,
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
            vae_to_transformer_projection = torch.nn.Conv2d(vae_latent_channels_actual, transformer_in_channels_actual, 1)
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
        # Debug: Print first raw example and columns
        try:
            logger.info(f"First raw example: {dataset[0]}")
        except Exception as e:
            logger.error(f"Could not print first raw example: {e}")
        logger.info(f"Dataset columns: {dataset.column_names}")
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
        try:
            logger.info(f"First raw example: {dataset[0]}")
        except Exception as e:
            logger.error(f"Could not print first raw example: {e}")
        logger.info(f"Loaded dataset with {len(dataset)} entries and columns: {dataset.column_names}")
        logger.info(f"First 5 raw examples: {[dataset[i] for i in range(min(5, len(dataset)))]}")
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
            remove_columns=list(set(original_columns) - columns_to_keep),
            desc="Running tokenizer on train dataset",
        )
        processed_dataset = processed_dataset.with_format(
            type="torch",
            columns=["pixel_values", "input_ids_2"],
        )
    elif args.dataset_type == "imagefolder":
        dataset = load_dataset("imagefolder", data_dir=args.dataset_path, split="train")
        # Debug: Print first raw example and columns
        try:
            logger.info(f"First raw example: {dataset[0]}")
        except Exception as e:
            pass
    # Debug: Print first preprocessed example before filtering
    try:
        logger.info(f"First preprocessed example: {processed_dataset[0]}")
    except Exception as e:
        logger.error(f"Could not print first preprocessed example: {e}")

    # --- Filter out invalid examples after preprocessing --- #
    def is_valid(example):
        pv = example["pixel_values"]
        if pv is None:
            return False
        # accept torch tensors *or* nested lists/arrays with at least 3 dims
        if isinstance(pv, torch.Tensor):
            return pv.ndim >= 3
        # treat lists as (C,H,W) flattened into one dimension per Arrow row
        return isinstance(pv, (list, tuple)) and len(pv) > 0    

    before_count = len(processed_dataset)
    logger.info(f"First 5 examples before filtering: {[processed_dataset[i] for i in range(min(5, before_count))]}")
    processed_dataset = processed_dataset.filter(is_valid)
    after_count = len(processed_dataset)
    logger.info(f"First 5 examples after filtering: {[processed_dataset[i] for i in range(min(5, after_count))]}")
    logger.info(f"Filtered dataset: {after_count} valid examples out of {before_count} after preprocessing.")

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

    # Fail fast if no training samples remain
    if len(train_dataset) == 0:
        raise RuntimeError(
            "All samples were filtered out – check dataset path, column names, or preprocessing."
        )

    # Collate function (Revert to simplest form)
    def collate_fn(examples):
        """Collates preprocessed examples into batches, filtering out invalid entries."""
        # Filter out entries where pixel_values is None (indicating a preprocessing failure for that example)
        valid_examples = [ex for ex in examples if ex["pixel_values"] is not None]

        if not valid_examples:
            return None

        try:
            batch = {
                "pixel_values": torch.stack([torch.as_tensor(ex["pixel_values"]) for ex in valid_examples]),
                "input_ids_2": torch.stack([torch.as_tensor(ex["input_ids_2"]) for ex in valid_examples]),
            }
            return batch
        except Exception as e:
            logger.error(f"Error during collate_fn stacking: {e}", exc_info=True)
            # Log tensor shapes for debugging if possible
            for i, ex in enumerate(valid_examples):
                pv_shape = ex['pixel_values'].shape if isinstance(ex.get('pixel_values'), torch.Tensor) else 'Not Tensor'
                id_shape = ex['input_ids_2'].shape if isinstance(ex.get('input_ids_2'), torch.Tensor) else 'Not Tensor'
                logger.error(f"  Example {i} shapes - pixel_values: {pv_shape}, input_ids_2: {id_shape}")
            return None # Skip batch if stacking fails

    import platform
    # Hugging Face warning: Pillow objects can't be pickled on Windows/macOS. Set num_proc=1 if not Linux.
    if platform.system() != "Linux":
        args.preprocessing_num_workers = 1
    # If fork unavailable (e.g., RunPod container), also set datasets.set_caching_enabled(False)
    try:
        import multiprocessing as mp
        if not hasattr(mp, "get_start_method") or mp.get_start_method() != "fork":
            datasets.set_caching_enabled(False)
            args.preprocessing_num_workers = 1
    except Exception:
        pass
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

    # --- Prepare with Accelerator ---
    # Prepare relevant items (handle val_dataloader conditionally)
    logger.info("Preparing models and dataloaders with Accelerator...")
    if val_dataloader:
        transformer, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
            transformer, optimizer, train_dataloader, val_dataloader
        )
    else:
        transformer, optimizer, train_dataloader = accelerator.prepare(
            transformer, optimizer, train_dataloader
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

    # --- Learning Rate Scheduler ---
    lr_scheduler = get_scheduler(
        "cosine", # Common scheduler type
        optimizer=optimizer,
        num_warmup_steps=50, # Example warmup steps
        num_training_steps=max_train_steps,
    )

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
                else:
                    logger.warning("Batch found with missing 'input_ids_2'. Handling as image-only.")
                    prompt_embeds_2 = None

                # Placeholder handling (Memory: 361714d3)
                if prompt_embeds_2 is None:
                    cross_attn_dim = transformer.config.cross_attention_dim
                    batch_size = pixel_values.shape[0]
                    prompt_embeds_2 = torch.zeros(
                        (batch_size, 1, cross_attn_dim),
                        dtype=weight_dtype, device=accelerator.device
                    )
                    logger.warning(f"Created null T5 embeds: {prompt_embeds_2.shape}")

                if input_ids_2_batch is None:
                    input_ids_2 = None 
                    logger.warning("Confirmed input_ids_2 is None for image-only batch.")
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

                    # [Bug 1.8] Ensure placeholder tensors use correct dtype (weight_dtype)
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
                img_ids = torch.arange(height * width, device=latents.device).unsqueeze(0)
                txt_ids = torch.zeros(1, dtype=torch.long, device=latents.device)
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
                # [Bug 1.2] Add noise to latents before predicting
                latents_noisy = noise_scheduler.add_noise(latents_reshaped, noise, timesteps)
                # [Bug 1.1] Use .sample from ModelOutput for loss
                model_pred = transformer(
                    hidden_states=latents_noisy,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds_2,
                    pooled_projections=clip_pooled,
                    img_ids=img_ids,
                    txt_ids=txt_ids
                ).sample
                target = noise
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                # Accumulate loss for epoch averaging
                train_loss += avg_loss.item()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # Clip gradients if needed (helps prevent exploding gradients)
                    accelerator.clip_grad_norm_(transformer.parameters(), 1.0)
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

                    # [Bug 1.3] Recompute latents, latents_reshaped, img_ids, noise for each val batch
                    latents_val = vae.encode(pixel_values_device).latent_dist.sample() * vae.config.scaling_factor
                    if vae_to_transformer_projection is not None:
                        latents_val = vae_to_transformer_projection(latents_val)
                    bsz_val, c_val, h_val, w_val = latents_val.shape
                    latents_reshaped_val = latents_val.permute(0, 2, 3, 1).reshape(bsz_val, h_val * w_val, c_val)
                    img_ids_val = torch.arange(h_val * w_val, device=latents_val.device).unsqueeze(0)
                    txt_ids_val = torch.zeros(1, dtype=torch.long, device=latents_val.device)
                    logger.debug(f"Generated validation 1D img_ids shape: {img_ids_val.shape}")

                    # Sample noise and timesteps for validation
                    noise_val = torch.randn_like(latents_reshaped_val)
                    timesteps_val = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz_val,), device=latents_val.device).long()

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

                    # [Bug 1.2] Add noise to latents for validation as well
                    latents_noisy_val = noise_scheduler.add_noise(latents_reshaped_val, noise_val, timesteps_val)
                    model_pred_val = transformer(
                        hidden_states=latents_noisy_val,
                        timestep=timesteps_val,
                        encoder_hidden_states=prompt_embeds_2_device, # T5 sequence embeds (None for img-only)
                        pooled_projections=clip_pooled_device, # CLIP pooled embeds (placeholder for img-only)
                        txt_ids=txt_ids_val, # Pass the variable prepared earlier (None for image-only)
                        img_ids=img_ids_val, # Use corrected 1D validation img_ids
                    ).sample

                    # Assume target is noise for validation loss calculation
                    target = noise_val
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