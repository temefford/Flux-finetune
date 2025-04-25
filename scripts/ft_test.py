import argparse
import math
import os
import time
from pathlib import Path

import datasets
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers import FluxPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from peft import LoraConfig, PeftModel
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

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
def tokenize_captions(captions, tokenizer):
    """Tokenizes captions using the provided tokenizer."""
    try:
        # FLUX uses a single T5 tokenizer for the second text input (input_ids_2)
        # Remove the assumption of unpacking two tokenizers
        # clip_tokenizer, t5_tokenizer = tokenizer # <-- REMOVE THIS LINE

        # Directly use the passed tokenizer (which should be tokenizer_2 from the pipeline)
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length, # Use the tokenizer's max length
            padding="max_length", # Pad to max length
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids # Return only the input_ids
    except Exception as e:
        logger.error(f"Error during caption tokenization: {e}")
        # Depending on desired behavior, either return None/empty tensor or raise
        raise e # Re-raise for now

def preprocess_train(examples, dataset_abs_path, image_transforms, image_column, hash_column, caption_column, tokenizer_2):
    """Preprocesses a batch of training examples."""
    try:
        # Append .jpg to the hash value retrieved using image_column
        # Use dataset_abs_path directly since images are not in 'imgs' subdir
        images = [Image.open(os.path.join(dataset_abs_path, f"{fn}.jpg")).convert("RGB") for fn in examples[image_column]]
        examples["pixel_values"] = [image_transforms(image) for image in images]
        
        # Tokenize caption column using tokenizer_2
        captions = list(examples[caption_column]) # Use caption_column
        # Ensure captions are strings
        captions = [str(c) if c is not None else "" for c in captions] # Handle potential None captions
        logger.debug(f"Tokenizing captions (first 5): {captions[:5]}")
        text_inputs_2 = tokenize_captions(captions, tokenizer_2) # Pass captions, use tokenizer_2
        examples["input_ids_2"] = text_inputs_2
    except FileNotFoundError as e:
        logger.error(f"Error opening image: {e}. Check dataset_path and file names.")
        # Decide how to handle: skip batch, raise error, etc.
        # For now, let's add a placeholder or skip - adding None might cause issues later
        # Safest might be to ensure paths are correct before this step.
        # Re-raising for now to make the error visible.
        raise e
    except Exception as e:
        logger.error(f"Error during image preprocessing: {e}")
        raise e
    return examples

# --- Preprocessing Function --- #
def preprocess_func(examples, **fn_kwargs):
    # Extract necessary variables passed via fn_kwargs
    image_transforms = fn_kwargs.get("image_transforms")
    tokenizer = fn_kwargs.get("tokenizer")
    tokenizer_2 = fn_kwargs.get("tokenizer_2")
    args = fn_kwargs.get("args")

    if not all([image_transforms, tokenizer, tokenizer_2, args]):
        raise ValueError("Missing required kwargs in preprocess_func (image_transforms, tokenizer, tokenizer_2, args)")

    # 1. Load and Transform Images
    # For 'imagefolder', the 'image' column contains PIL Image objects
    # For 'hf_metadata', it might contain paths or bytes - adjust loading if needed
    try:
        images = [image.convert("RGB") for image in examples[args.image_column]]
    except Exception as e:
        logger.error(f"Error accessing/converting image column '{args.image_column}': {e}")
        logger.error(f"Example keys: {examples.keys()}")
        # Handle potential issues like incorrect column name or non-image data
        # For now, let's re-raise or return an empty dict to signal failure
        raise e # Or handle more gracefully

    pixel_values = torch.stack([image_transforms(image) for image in images])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    # 2. Tokenize Captions (Handle 'imagefolder' case)
    if args.dataset_type == "imagefolder":
        # Generate dummy captions and add them directly to the examples dict
        num_examples = len(images) # Use length of successfully loaded images
        examples[args.caption_column] = ["" for _ in range(num_examples)] # Add dummy captions under the expected key
        # Now call tokenize_captions with the modified examples dict
        captions = tokenize_captions((tokenizer, tokenizer_2), examples, text_column=args.caption_column)
    elif args.dataset_type == "hf_metadata": # Assuming hf_metadata has the caption column
        # Check if caption column exists before accessing
        if args.caption_column not in examples:
            logger.error(f"Caption column '{args.caption_column}' not found in 'hf_metadata' dataset example.")
            logger.error(f"Available columns: {list(examples.keys())}")
            # Decide how to handle: raise error, skip example, use empty caption?
            raise KeyError(f"Caption column '{args.caption_column}' missing for hf_metadata.")
        captions = tokenize_captions((tokenizer, tokenizer_2), examples, text_column=args.caption_column)
    else:
        # Should not happen if initial loading logic is correct, but good to handle
        logger.error(f"Preprocessing encountered unexpected dataset type: {args.dataset_type}")
        # Fallback or raise error
        raise ValueError(f"Unsupported dataset type in preprocess_func: {args.dataset_type}")

    # 3. VAE Encoding (if not done in training loop)
    # Moved VAE encoding to the training loop for efficiency
    # latents = vae.encode(pixel_values.to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
    # latents = latents * vae.config.scaling_factor

    # Return processed data
    return {
        "pixel_values": pixel_values,
        "input_ids": captions["input_ids"],
        "input_ids_2": captions["input_ids_2"],
        "attention_mask": captions["attention_mask"],
        "attention_mask_2": captions["attention_mask_2"],
    }

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

    # Define image transformations
    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.image_resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.image_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # Standard normalization for [-1, 1] range
        ]
    )

    logger.info("Preprocessing training data...")
    # Ensure dataset path is absolute for image loading
    dataset_abs_path = os.path.abspath(args.dataset_path)
    preprocess_kwargs = {
        "image_transforms": image_transforms,
        "tokenizer_2": tokenizer_2,
        "dataset_abs_path": dataset_abs_path,        # Added back
        "image_column": args.image_column,           # Added back
        "hash_column": args.image_column,            # Added back (using image_column value based on config)
        "caption_column": args.caption_column         # Added back
    }
    train_dataset = train_dataset.map(
        preprocess_train,
        batched=True,
        num_proc=1, # Changed from args.preprocessing_num_workers
        remove_columns=train_dataset.column_names,
        fn_kwargs=preprocess_kwargs # Pass variables here
    )
    # Apply preprocessing to val dataset if it exists
    if val_dataset:
        logger.info("Preprocessing validation data...")
        # Ensure dataset path is absolute for image loading
        dataset_abs_path_val = os.path.abspath(args.dataset_path) # Recalculate or reuse dataset_abs_path
        preprocess_kwargs_val = {
            "image_transforms": image_transforms,
            "tokenizer_2": tokenizer_2,
            "dataset_abs_path": dataset_abs_path_val,   # Added back
            "image_column": args.image_column,          # Added back
            "hash_column": args.image_column,           # Added back (using image_column value based on config)
            "caption_column": args.caption_column        # Added back
        }
        val_dataset = val_dataset.map(
            preprocess_train, # Use the same function
            batched=True,
            num_proc=1, # Changed from args.preprocessing_num_workers
            remove_columns=val_dataset.column_names,
            fn_kwargs=preprocess_kwargs_val # Pass variables here too
        )

    # Collate function
    def collate_fn(examples):
        pixel_values = torch.stack([torch.tensor(example["pixel_values"]) for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = torch.stack([torch.tensor(example["input_ids"]) for example in examples])
        attention_mask = torch.stack([torch.tensor(example["attention_mask"]) for example in examples])

        batch = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Add text_encoder_2 inputs if they exist in the batch
        if 'input_ids_2' in examples[0]:
            input_ids_2 = torch.stack([torch.tensor(example["input_ids_2"]) for example in examples])
            attention_mask_2 = torch.stack([torch.tensor(example["attention_mask_2"]) for example in examples])
            batch["input_ids_2"] = input_ids_2
            batch["attention_mask_2"] = attention_mask_2

        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers, # Use arg for num_workers
    )

    # Create validation dataloader if val_dataset exists
    val_dataloader = None
    if val_dataset:
        logger.info("Creating validation dataloader...")
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
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
    if val_dataset: # Check if val_dataset was created before logging its length
        logger.info(f"  Num validation examples = {len(val_dataset)}")
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
                input_ids_batch = batch["input_ids"].to(accelerator.device) if "input_ids" in batch and batch["input_ids"] is not None else None
                input_ids_2_batch = batch["input_ids_2"].to(accelerator.device) if "input_ids_2" in batch and batch["input_ids_2"] is not None else None

                if input_ids_batch is not None and input_ids_2_batch is not None:
                    input_ids = input_ids_batch # Assign to outer scope var
                    input_ids_2 = input_ids_2_batch # Assign to outer scope var
                    with torch.no_grad(): # Text encoding should not require gradients
                        prompt_embeds_outputs = text_encoder(input_ids, output_hidden_states=True)
                        clip_pooled = prompt_embeds_outputs.pooler_output
                        prompt_embeds_2_outputs = text_encoder_2(input_ids_2, output_hidden_states=True)
                        prompt_embeds_2 = prompt_embeds_2_outputs.last_hidden_state
                    logger.debug("Encoded text inputs for multimodal batch.")
                else:
                     logger.warning("Missing text inputs for non-imagefolder dataset batch! Using placeholders.")
                     # Handle error or create placeholders if appropriate
                     clip_embed_dim = text_encoder.config.projection_dim   # e.g., 1280
                     clip_pooled = torch.zeros(batch_size, clip_embed_dim, dtype=weight_dtype, device=accelerator.device)
                     logger.debug(f"Created placeholder clip_pooled: {clip_pooled.shape}")
                     # Keep text IDs/embeds as None

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

                # Create null T5 input IDs if still None
                if input_ids_2 is None:
                    t5_pad_token_id = tokenizer_2.pad_token_id if hasattr(tokenizer_2, 'pad_token_id') and tokenizer_2.pad_token_id is not None else 0
                    input_ids_2 = torch.full(
                        (batch_size, null_sequence_length),
                        fill_value=t5_pad_token_id,
                        dtype=torch.long, device=accelerator.device
                    )
                    logger.debug(f"Created null T5 input IDs: {input_ids_2.shape}")
                else:
                    input_ids_2 = input_ids_2.long()
            # For imagefolder dataset, we intentionally leave prompt_embeds_2 and input_ids_2 as None so that
            # later logic can create placeholders matching the image token sequence length (seq_len).
             # --- End Unconditional Handling --- #

            # --- VAE Encoding and Noise Addition (Common Logic) ---
            with accelerator.accumulate(transformer): # Use transformer here
                # Encode pixel values -> latents using VAE
                # VAE is prepared by accelerator, handles device/dtype internally via hooks/casting
                latents = vae.encode(pixel_values).latent_dist.sample()
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
                    prompt_embeds_2 = torch.zeros(bsz, 1, cross_attention_dim, dtype=weight_dtype, device=accelerator.device)
                    logger.warning(f"prompt_embeds_2 was None, created placeholder: {prompt_embeds_2.shape}")

                # Store original input_ids_2 state for conditional logic
                original_input_ids_2_is_none = input_ids_2 is None

                # Ensure input_ids_2 is None for image-only batches before the call
                if input_ids_2 is None:
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

                # Revert to creating a minimal placeholder for input_ids_2
                if original_input_ids_2_is_none:
                     input_ids_2 = torch.zeros(bsz, 1, dtype=torch.long, device=accelerator.device)
                     logger.warning(f"input_ids_2 was None, created minimal placeholder: {input_ids_2.shape}")

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
        if val_dataloader:
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
            if val_dataloader:
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
        if val_dataloader:
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
