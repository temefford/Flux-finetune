import argparse
import math
import os
from functools import partial
from pathlib import Path
import time

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
import json

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

    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Maximum number of checkpoints to keep.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging and potentially slower operations.")
    parser.add_argument("--text_ids_json_path", type=str, default=None, help="Optional path to a JSON file mapping image hashes to text_ids.")
    parser.add_argument("--text_ids_column", type=str, default=None, help="Column name for text_ids if using --text_ids_json_path.")

    cmd_args = parser.parse_args() # Parse command line args first

    # --- Load Configuration from YAML if specified --- #
    config_args = {} # Dictionary to hold args from config file
    if cmd_args.config:
        try:
            with open(cmd_args.config, 'r') as f:
                config = yaml.safe_load(f)
                if config: # Ensure config is not empty
                    config_args = config
        except FileNotFoundError:
            print(f"ERROR: Configuration file not found at: {cmd_args.config}")
            raise
        except Exception as e:
            print(f"ERROR: Error loading configuration file {cmd_args.config}: {e}")
            raise

    # --- Merge Arguments: Priority: Command Line > Config File > Parser Defaults --- #

    # 1. Get parser defaults
    parser_defaults = parser.parse_args([])
    final_args = argparse.Namespace(**vars(parser_defaults)) # Start with defaults

    # 2. Update with config file arguments
    for k, v in config_args.items():
        if hasattr(final_args, k): # Only update if it's a known argument
            setattr(final_args, k, v)
        else:
            # Use print for warnings during arg parsing before logger is ready
            print(f"WARNING: Ignoring unknown argument '{k}' from config file '{cmd_args.config}'")

    # 3. Update with command line arguments (highest priority)
    # Iterate through args defined in the parser to only consider known args
    for k in vars(parser_defaults):
        cmd_val = getattr(cmd_args, k, None)
        # Check if the command line value is different from the parser's default
        # This handles boolean flags correctly (e.g., --debug without being in config)
        # and ensures cmd line explicitly set values (even if None) override config/defaults
        if cmd_val != getattr(parser_defaults, k):
            setattr(final_args, k, cmd_val)

    # Store the config path if one was used
    final_args.config_path = cmd_args.config

    args = final_args # Assign the final merged args

    # Add any derived arguments or defaults needed after merging
    args.train_batch_size = getattr(args, 'train_batch_size', 4)

    # --- Post-merge adjustments specific to certain args ---
    # Use data_dir as dataset_path if dataset_path wasn't set by config or cmd line override
    if hasattr(args, 'data_dir') and not hasattr(args, 'dataset_path'):
         args.dataset_path = args.data_dir
         print(f"INFO: Setting dataset_path from data_dir: {args.dataset_path}")

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
    args.validation_batch_size = getattr(args, 'validation_batch_size', args.train_batch_size)
    args.dataloader_num_workers = getattr(args, 'dataloader_num_workers', 0)
    # Ensure validation_split has a default value if not set anywhere
    if not hasattr(args, 'validation_split'):
        args.validation_split = 0.0 # Default to 0 if not in config or cmd line
        print("WARNING: validation_split not found in config or command line, defaulting to 0.0") # Use print

    print("--- Final Arguments before return ---")
    print(vars(args))
    print("-------------------------------------")
    return args

# --- New Transform Function for set_transform --- #
def transform_example(batch, dataset_abs_path, image_transforms, image_column, caption_column, hash_column, tokenizer_2, text_id_map=None):
    # Use the global logger instance configured in main
    logger = logging.getLogger(__name__)

    # Determine batch size
    # Get the list for one of the columns (e.g., hash_column or caption_column)
    # Need to handle cases where hash or caption column might not be present
    if hash_column and hash_column in batch:
        batch_size = len(batch[hash_column])
    elif caption_column and caption_column in batch:
        batch_size = len(batch[caption_column])
    else:
        # Try to infer from another column, this is less robust
        try:
            first_key = list(batch.keys())[0]
            batch_size = len(batch[first_key])
        except (IndexError, TypeError):
             logger.error("Could not determine batch size from input batch dictionary.")
             # Return empty dictionary or raise error? Returning empty for now.
             return {"pixel_values": [], "input_ids_2": [], "text_ids": []}

    # Initialize lists to store processed outputs
    pixel_values_list = []
    input_ids_2_list = []
    text_ids_list = []
    valid_indices = [] # Keep track of indices that were processed successfully

    for i in range(batch_size):
        image_hash = None
        caption = ""
        current_text_ids = None

        try:
            # --- Extract data for the current item --- #
            if hash_column and hash_column in batch:
                image_hash = batch[hash_column][i]
            if caption_column and caption_column in batch:
                caption = batch[caption_column][i]
                if caption is None:
                    caption = ""
                elif not isinstance(caption, str):
                    caption = str(caption)

            # Basic validation
            if not isinstance(image_hash, str) or not image_hash:
                logger.warning(f"Invalid or missing image hash at index {i} (value: {repr(image_hash)}). Skipping item.")
                continue

            # --- Process the single item --- #
            path = os.path.join(dataset_abs_path, image_hash + ".jpg") # Assuming jpg

            # 1. Load Image
            image = Image.open(path).convert("RGB")

            # 2. Apply Image Transformations (as a batch of 1)
            image_input_dict = image_transforms([image])
            pixel_values_tensor = image_input_dict['pixel_values']
            if isinstance(pixel_values_tensor, np.ndarray):
                pixel_values_tensor = torch.from_numpy(pixel_values_tensor)
            pixel_values_tensor = pixel_values_tensor.squeeze(0) # Remove batch dim

            # 3. Tokenize Caption
            max_len = getattr(tokenizer_2, 'model_max_length', 512)
            text_input_dict = tokenizer_2(
                caption, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt"
            )
            input_ids_2_tensor = text_input_dict['input_ids'].squeeze(0) # Remove batch dimension

            # 4. Get Text IDs (optional)
            if text_id_map is not None:
                text_ids_val = text_id_map.get(image_hash, None)
                if text_ids_val is not None:
                    try:
                        if isinstance(text_ids_val, (int, float)):
                            current_text_ids = torch.tensor([text_ids_val], dtype=torch.long)
                        elif isinstance(text_ids_val, list):
                            current_text_ids = torch.tensor(text_ids_val, dtype=torch.long)
                        # Add handling for tensor if needed
                        # elif isinstance(text_ids_val, torch.Tensor):
                        #     current_text_ids = text_ids_val.to(dtype=torch.long)
                        else:
                            logger.warning(f"Unsupported type for text_ids_val {type(text_ids_val)} for hash {image_hash}. Skipping text_ids.")
                    except Exception as e:
                         logger.error(f"Error converting text_ids '{text_ids_val}' for hash {image_hash}: {e}. Skipping.")
            # If lookup failed or map is None, current_text_ids remains None

            # --- Append successful results --- #
            pixel_values_list.append(pixel_values_tensor)
            input_ids_2_list.append(input_ids_2_tensor)
            # Only append text_ids if they were successfully processed
            if current_text_ids is not None:
                 text_ids_list.append(current_text_ids)
            # Keep track of successful index
            valid_indices.append(i)

        except FileNotFoundError:
            logger.warning(f"Image file not found for hash {image_hash} at index {i}: {path}. Skipping item.")
        except Exception as e:
            logger.error(f"Error processing item at index {i} (hash: {image_hash}): {e}", exc_info=True)
            # Continue to next item in batch

    # --- Construct the final batch dictionary --- #
    output_batch = {}
    # Initialize with lists of Nones matching the input batch size
    output_batch["pixel_values"] = [None] * batch_size
    output_batch["input_ids_2"] = [None] * batch_size
    # Conditionally initialize text_ids column only if it was requested
    if text_id_map is not None:
        output_batch["text_ids"] = [None] * batch_size

    processed_count = 0
    text_ids_processed_count = 0

    # Fill the lists with actual data for successfully processed items
    for idx, original_index in enumerate(valid_indices):
        output_batch["pixel_values"][original_index] = pixel_values_list[idx]
        output_batch["input_ids_2"][original_index] = input_ids_2_list[idx]
        processed_count += 1
        # Check if text_ids were processed for this valid item
        # Ensure text_ids_list has an entry for this processed item
        if text_id_map is not None and len(text_ids_list) > idx and text_ids_list[idx] is not None:
             if "text_ids" in output_batch:
                output_batch["text_ids"][original_index] = text_ids_list[idx]
                text_ids_processed_count += 1

    if processed_count == 0:
        logger.warning("No items processed successfully in the batch.")
        # The batch dictionary already contains lists of Nones

    # Decide whether to keep the text_ids column in the output batch
    # We only need it if the map was provided AND at least one ID was successfully processed.
    keep_text_ids_col = text_id_map is not None and text_ids_processed_count > 0

    if not keep_text_ids_col and "text_ids" in output_batch:
        # Remove the column if no valid text IDs were found or if map wasn't provided.
        # This prevents sending a column of all Nones downstream.
        del output_batch["text_ids"]
    elif text_id_map is not None and processed_count > 0 and text_ids_processed_count < processed_count:
        # Log if some text_ids were found but not for all successfully processed items
        logger.debug(f"Processed {text_ids_processed_count} text_ids out of {processed_count} successfully processed items. Missing items will have None.")

    # Optional: Check if essential columns ended up being all None (should only happen if processed_count == 0)
    # if all(v is None for v in output_batch["pixel_values"]):
    #     logger.warning("pixel_values list contains only None after processing the batch.")
    # if all(v is None for v in output_batch["input_ids_2"]):
    #     logger.warning("input_ids_2 list contains only None after processing the batch.")

    return output_batch

# --- End New Transform Function --- #

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
        _transform_func = partial(
            transform_example, # Use the NEW function
            dataset_abs_path=args.data_dir,
            image_transforms=pipeline.image_processor.preprocess,
            image_column=args.image_column,
            caption_column=args.caption_column,
            hash_column=getattr(args, 'hash_column', None), # Pass hash_column if defined, else None
            tokenizer_2=tokenizer_2,
        )

        logger.info("Preprocessing dataset...")
        processed_dataset = dataset.map(
            _transform_func,
            batched=True,
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

        _transform_func = partial(
             transform_example, # Use the NEW function
             dataset_abs_path=args.data_dir,
             image_transforms=pipeline.image_processor.preprocess,
             image_column=args.image_column,
             caption_column=None, # Not needed for imagefolder
             hash_column=None, # Not needed for imagefolder
             tokenizer_2=tokenizer_2, # Still needed to create dummy text inputs
             text_id_map=None # No text IDs for imagefolder
        )
        processed_dataset = dataset.map(
             _transform_func,
             batched=True,
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

    # Load text_ids if provided
    text_id_map = None
    if args.text_ids_json_path:
        # Ensure text_id_map is loaded *before* setting the transform
        try:
            with open(args.text_ids_json_path, 'r') as f:
                text_id_map = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load text_id_map from {args.text_ids_json_path}: {e}")
            return

    # --- Set Transform for Lazy Processing --- #
    # Define the transform function with necessary arguments fixed
    _transform_func = partial(
        transform_example, # Use the NEW function
        dataset_abs_path=args.data_dir,
        image_transforms=pipeline.image_processor.preprocess,
        image_column=args.image_column,
        caption_column=args.caption_column,
        hash_column=args.hash_column,
        tokenizer_2=tokenizer_2,
        text_id_map=text_id_map # Pass the loaded map
    )

    # Define columns expected by the transform function (input columns)
    transform_input_columns = [args.hash_column, args.caption_column]
    if args.text_ids_json_path:
        transform_input_columns.append(args.text_ids_column)
    # For ImageFolder, the input is just 'image'
    if args.dataset_type == "imagefolder":
        transform_input_columns = [args.image_column]

    logger.info(f"Setting transform function for dataset (Input columns: {transform_input_columns})")
    train_dataset.set_transform(_transform_func, columns=transform_input_columns)
    if eval_dataset:
        eval_dataset.set_transform(_transform_func, columns=transform_input_columns)

    # --- DataLoader --- #
    # Define collate_fn before DataLoader
    def collate_fn(examples, expected_clip_dim=768, cross_attention_dim=4096):
        logger = logging.getLogger("collate_fn")
        # logger.setLevel(logging.DEBUG) # Optional: Set specific level for this logger
        logger.debug(f"Received {len(examples)} examples in collate_fn")

        # 1. Filter out None examples (where transform_example failed)
        valid_examples = []
        for i, ex in enumerate(examples):
            if ex is not None and isinstance(ex, dict) and ex.get("pixel_values") is not None and ex.get("input_ids_2") is not None:
                valid_examples.append(ex)
            else:
                logger.debug(f"  Discarding example at index {i}. Content: {ex}")

        if not valid_examples:
            logger.warning("collate_fn: No valid examples found in the batch after filtering. Returning None.")
            return None

        # 2. Extract and stack tensors
        try:
            pixel_values = torch.stack([ex["pixel_values"] for ex in valid_examples])
            input_ids_2 = torch.stack([ex["input_ids_2"] for ex in valid_examples])

            batch = {
                "pixel_values": pixel_values,
                "input_ids_2": input_ids_2,
            }

            # Check if *any* example actually had non-padding text tokens (more robust check for 'has_text')
            # Assuming padding token ID is 0 for tokenizer_2
            # This check might need adjustment based on the actual tokenizer's padding ID
            # A simpler check is if captions were present and non-empty during transform_example
            # But since input_ids_2 is always generated (even from empty string), we check its content.
            # For now, we'll rely on the presence of the key, assuming transform_example put valid data.
            # If input_ids_2 could be all padding for image-only, we need a better flag from transform.
            # Let's start simple: if input_ids_2 key exists, assume text is intended.
            batch["has_text"] = True # Assume text is present if input_ids_2 is successfully stacked

            # Handle optional text_ids
            if all("text_ids" in ex for ex in valid_examples):
                try:
                    text_ids = torch.stack([ex["text_ids"] for ex in valid_examples])
                    batch["text_ids"] = text_ids
                    # If text_ids are present, we definitely have text-related info
                    batch["has_text"] = True
                except Exception as stack_err:
                    logger.error(f"Error stacking 'text_ids': {stack_err}. Skipping 'text_ids' for this batch.")
                    # Fall through without text_ids
            else:
                logger.debug("Not all valid examples in the batch have 'text_ids'. 'text_ids' key will be absent.")
                # If input_ids_2 was present but text_ids wasn't, still consider it has_text=True based on input_ids_2

        except Exception as e:
            logger.error(f"Error during tensor stacking in collate_fn: {e}", exc_info=True)
            logger.error(f"Problematic valid_examples sample: {valid_examples[:2]}") # Log sample data
            return None # Signal error to DataLoader

        # If stacking pixel_values succeeded but input_ids_2 failed or wasn't present in *any* valid example
        # (This case is less likely with current transform_example which always creates input_ids_2)
        if "pixel_values" in batch and "input_ids_2" not in batch:
            logger.debug("Batch contains pixel_values but not input_ids_2. Marking as image-only.")
            batch["has_text"] = False
        elif "pixel_values" not in batch:
            logger.warning("Collate function resulted in a batch without pixel_values. Returning None.")
            return None # Cannot proceed without images

        logger.debug(f"Collate function returning batch with keys: {list(batch.keys())}")
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=0, # args.dataloader_num_workers,
    )
    logger.info("DataLoader instantiation complete.")

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

                # Prepare conditional inputs based on whether text is present
                transformer_kwargs = {
                    "img_latent": latents_reshaped,
                    "timesteps": timesteps,
                }

                if batch.get("has_text", False):
                    # --- Text-Conditional Batch --- #
                    # Required: txt_ids, pooled_projections, encoder_hidden_states
                    transformer_kwargs["txt_ids"] = batch["input_ids_2"]

                    # TODO: Get actual pooled_projections and encoder_hidden_states
                    # These should ideally come from the CLIP text encoders used by Flux.
                    # For now, using placeholders as before, but only for text batches.
                    clip_dim = getattr(transformer.config, 'projection_dim', 768) # Get from config if possible
                    cross_attn_dim = getattr(transformer.config, 'cross_attention_dim', 4096)
                    transformer_kwargs["pooled_projections"] = torch.zeros(bsz, clip_dim, dtype=weight_dtype, device=accelerator.device)
                    transformer_kwargs["encoder_hidden_states"] = torch.zeros(bsz, 1, cross_attn_dim, dtype=weight_dtype, device=accelerator.device)
                    logger.debug("Running text-conditional forward pass (using text placeholders)")

                else:
                    # --- Image-Only Batch --- #
                    # Required: Placeholders for pooled_projections and encoder_hidden_states
                    # DO NOT pass txt_ids
                    clip_dim = getattr(transformer.config, 'projection_dim', 768)
                    cross_attn_dim = getattr(transformer.config, 'cross_attention_dim', 4096)
                    transformer_kwargs["pooled_projections"] = torch.zeros(bsz, clip_dim, dtype=weight_dtype, device=accelerator.device)
                    transformer_kwargs["encoder_hidden_states"] = torch.zeros(bsz, 1, cross_attn_dim, dtype=weight_dtype, device=accelerator.device)
                    logger.debug("Running image-only forward pass (using placeholders)")

                # Call the transformer with dynamically built kwargs
                model_pred = transformer(**transformer_kwargs).prediction

                # Calculate loss (compare model prediction to noise)
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

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

                    # Ensure pooled projections exist (create placeholder if needed) -- Moved here AGAIN to prevent logging error
                    if clip_pooled_device is None:
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

                        clip_pooled_device = torch.zeros(bsz, expected_clip_dim, dtype=weight_dtype, device=accelerator.device)
                        logger.warning(f"clip_pooled was None immediately before logging/transformer call, created placeholder with expected dim {expected_clip_dim}: {clip_pooled_device.shape}")

                    # Ensure prompt_embeds_2 is a tensor, even for image-only datasets
                    if prompt_embeds_2_device is None:
                        cross_attention_dim = getattr(transformer.config, 'cross_attention_dim', None)
                        if cross_attention_dim is None:
                            cross_attention_dim = getattr(transformer.config, 'joint_attention_dim', None)
                        if cross_attention_dim is None:
                            raise ValueError("Could not determine cross_attention_dim for placeholder prompt_embeds_2.")
                        # Use seq_len=1 for null prompt
                        prompt_embeds_2_device = torch.zeros(
                            bsz, 1, cross_attention_dim,
                            dtype=weight_dtype, device=accelerator.device
                        )
                        logger.warning(f"prompt_embeds_2 was None, created placeholder: {prompt_embeds_2_device.shape}")

                    # Ensure input_ids_2 is None for image-only batches before the call
                    if input_ids_2_device is None:
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

                    # Predict noise using the model
                    # Pass the prepared conditional inputs (placeholders or None)
                    logger.debug(f"  transformer input shape - hidden_states: {latents_reshaped.shape}")
                    logger.debug(f"  transformer input shape - timestep: {timesteps.shape}") 
                    logger.debug(f"  transformer input shape - encoder_hidden_states: {prompt_embeds_2_device.shape}") # Placeholder ensured
                    logger.debug(f"  transformer input shape - pooled_projections: {clip_pooled_device.shape}")       # Placeholder ensured
                    logger.debug(f"  transformer input shape - img_ids: {img_ids_val.shape}")
                    # Log txt_ids shape only if it exists 
                    if input_ids_2 is not None:
                         logger.debug(f"  transformer input shape - txt_ids: {input_ids_2.shape}")
                    else:
                         logger.debug("  transformer input shape - txt_ids: None") # Explicitly logging None

                    # Pass arguments explicitly
                    model_pred_val = transformer(
                        hidden_states=latents_reshaped,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds_2_device,
                        pooled_projections=clip_pooled_device,
                        txt_ids=input_ids_2, # Pass the placeholder tensor
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
    # --- Argument Parsing --- #
    args = parse_args()
 
    # --- Main Execution --- #
    main(args)
