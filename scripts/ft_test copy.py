import argparse
import logging
import math
import os
import time
from pathlib import Path

import datasets
import diffusers
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers import FluxPipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig, PeftModel
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

logger = get_logger(__name__, log_level="INFO")

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a FLUX training script.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ft_config.yaml",
        help="Path to the configuration YAML file relative to the project root.",
    )
    # Add arguments for data_dir and output_dir to potentially override config
    parser.add_argument("--data_dir", type=str, default=None, help="Override dataset path from config.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory from config.")

    cmd_args = parser.parse_args() # Parse command line args first

    # Load config from YAML file
    try:
        with open(cmd_args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {cmd_args.config}")
        raise
    except Exception as e:
        logging.error(f"Error loading configuration file {cmd_args.config}: {e}")
        raise

    # Create Namespace from YAML config
    args = argparse.Namespace(**config)

    # Apply command-line overrides (if provided)
    # Logging will happen in main() after accelerator is initialized
    if cmd_args.data_dir:
        args.dataset_path = cmd_args.data_dir # Note: config key is dataset_path
    if cmd_args.output_dir:
        args.output_dir = cmd_args.output_dir

    # --- Path Adjustments --- 
    # Make output_dir relative to the project root (where the script is run from)
    # Assuming the script is run from the 'finetuning' directory
    # No adjustment needed if output_dir is already relative like 'outputs'
    # If output_dir was specified as absolute in YAML or command line, it stays absolute.
    # Let's ensure it exists
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Ensure dataset_path is treated as absolute if needed (e.g., /workspace/art)
    # The config expects an absolute path, so no adjustment needed unless overridden
    if not hasattr(args, 'dataset_path') or not args.dataset_path:
         raise ValueError("dataset_path must be specified in the config file or via --data_dir")

    # Add other defaults if missing from config
    args.config_path = cmd_args.config # Store the actual config path used
    args.validation_batch_size = getattr(args, 'validation_batch_size', args.batch_size)
    args.dataloader_num_workers = getattr(args, 'dataloader_num_workers', 4)
    args.preprocessing_num_workers = getattr(args, 'preprocessing_num_workers', 1)

    return args

# --- Data Preprocessing Helper Functions ---
def tokenize_captions(tokenizer, examples, text_column="text"):
    """Tokenizes captions from the specified text column using both CLIP and T5 tokenizers.

    Args:
        tokenizer: The tokenizer instance (should be a tuple of CLIP and T5 tokenizers for FLUX).
        examples: A dictionary-like object containing the data batch.
        text_column (str): The name of the column containing the text captions.

    Returns:
        A dictionary containing tokenized 'input_ids', 'attention_mask', etc.
    """
    clip_tokenizer, t5_tokenizer = tokenizer

    captions = []
    for caption in examples[text_column]: # Use text_column argument
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, tuple)):
            # Handle lists/tuples if necessary, for now just take the first element
            # Or adapt logic based on expected format (e.g., random.choice for training)
            captions.append(caption[0] if caption else "") # Fallback for empty list/tuple
        else:
            captions.append(str(caption)) # Convert other types to string

    # Tokenize with CLIP tokenizer
    text_inputs = clip_tokenizer(
        captions, max_length=clip_tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    input_ids = text_inputs.input_ids
    attention_mask = text_inputs.attention_mask

    # Tokenize with T5 tokenizer
    text_inputs_2 = t5_tokenizer(
        captions, max_length=t5_tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    input_ids_2 = text_inputs_2.input_ids
    attention_mask_2 = text_inputs_2.attention_mask

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "input_ids_2": input_ids_2,
        "attention_mask_2": attention_mask_2,
    }

def preprocess_train(examples, dataset_abs_path, image_transforms, image_column):
    """Preprocesses a batch of training examples."""
    try:
        image_dir = os.path.join(dataset_abs_path, "imgs")
        # Append .jpg to the hash value retrieved using image_column
        images = [Image.open(os.path.join(image_dir, f"{fn}.jpg")).convert("RGB") for fn in examples[image_column]]
        examples["pixel_values"] = [image_transforms(image) for image in images]
    except FileNotFoundError as e:
        logging.error(f"Error opening image: {e}. Check dataset_path and file names.")
        # Decide how to handle: skip batch, raise error, etc.
        # For now, let's add a placeholder or skip - adding None might cause issues later
        # Safest might be to ensure paths are correct before this step.
        # Re-raising for now to make the error visible.
        raise e
    except Exception as e:
        logging.error(f"Error during image preprocessing: {e}")
        raise e
    return examples

# --- Main Function ---
def main(args):
    logging_dir = Path(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard", # or "wandb" if configured
        project_config=accelerator_project_config,
    )

    # Log overrides now that accelerator is initialized
    if hasattr(args, 'config_path'): # Check if config was successfully loaded
        if args.config_path != 'configs/ft_config.yaml': # Log if not default
             logger.info(f"Using configuration file: {args.config_path}", main_process_only=True)
        # Check if paths were overridden (by comparing with originally loaded values if we stored them, or just checking if cmd_args existed - easier)
        # Simplest: Check if the override args were passed (we need to get cmd_args again or pass them)
        # Let's re-parse minimal args just for checking overrides
        parser = argparse.ArgumentParser(add_help=False) # Don't add help again
        parser.add_argument("--data_dir", type=str, default=None)
        parser.add_argument("--output_dir", type=str, default=None)
        cmd_override_args, _ = parser.parse_known_args() # Parse only known args

        if cmd_override_args.data_dir:
            logger.info(f"Overriding dataset_path with command-line value: {args.dataset_path}", main_process_only=True)
        if cmd_override_args.output_dir:
             logger.info(f"Overriding output_dir with command-line value: {args.output_dir}", main_process_only=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

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
        noise_scheduler = pipeline.scheduler
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
 
    if vae_latent_channels_actual != transformer_in_channels_actual:
        logger.warning(
            f"Runtime shape mismatch detected: VAE output {vae_latent_channels_actual} channels, "
            f"Transformer input {transformer_in_channels_actual} channels. Adding projection layer {vae_latent_channels_actual}->{transformer_in_channels_actual}."
        )
        vae_to_transformer_projection = torch.nn.Linear(vae_latent_channels_actual, transformer_in_channels_actual)
        vae_to_transformer_projection.to(accelerator.device, dtype=weight_dtype)
    else:
        # This case seems unlikely given the error, but included for completeness
        logger.info("VAE output and Transformer input channels match according to configs.")
        vae_to_transformer_projection = None
 
    # --- Optimizer Setup --- 
    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    if vae_to_transformer_projection is not None:
        # Ensure projection layer params require grad and add them
        for param in vae_to_transformer_projection.parameters():
            param.requires_grad = True 
        params_to_optimize.extend(vae_to_transformer_projection.parameters())
        logger.info("Added VAE->Transformer projection layer parameters to optimizer.")

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=float(args.learning_rate), # Ensure learning_rate is float
        betas=(0.9, 0.999),
        weight_decay=1e-2, # Common weight decay
        eps=1e-08,
    )

    # --- Dataset Loading and Preprocessing ---
    # Calculate absolute dataset path relative to the script's location
    script_dir = os.path.dirname(__file__)
    dataset_abs_path = os.path.abspath(os.path.join(script_dir, args.dataset_path))
    metadata_file = os.path.join(dataset_abs_path, "metadata.json") # Use absolute path for metadata too
    logger.info(f"Attempting to load dataset metadata from: {metadata_file}")

    if not os.path.exists(metadata_file):
        logger.error(f"Metadata file not found at {metadata_file}")
        logger.error("Ensure 'metadata.json' exists in the specified dataset_path.")
        return

    try:
        # Load as standard JSON (list of dicts at root)
        dataset = load_dataset("json", data_files=metadata_file, split="train") # Removed field="data"
        logger.info(f"Loaded dataset with {len(dataset)} examples.")
    except Exception as e:
        logger.error(f"Failed to load dataset from {metadata_file}: {e}")
        logger.error("Ensure 'metadata.json' exists and is formatted correctly (JSON array of objects with 'file_name' and 'text').")
        logger.error("Alternatively, modify the 'load_dataset' call for your structure.")
        return

    # --- Split Dataset --- #
    if args.val_split > 0.0:
        if not (0 < args.val_split < 1):
            raise ValueError("val_split must be between 0 and 1 (exclusive)")
        logger.info(f"Splitting dataset with validation split: {args.val_split}")
        # Use the datasets library's built-in splitting method
        split_dataset = dataset.train_test_split(test_size=args.val_split, seed=args.seed)
        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]
        logger.info(f"  Training samples: {len(train_dataset)}")
        logger.info(f"  Validation samples: {len(val_dataset)}")
    else:
        train_dataset = dataset
        val_dataset = None
        logger.info(f"Using full dataset for training: {len(train_dataset)} samples.")

    # Define image transformations
    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.image_resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.image_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # Standard normalization for [-1, 1] range
        ]
    )

    # Define the preprocessing function with necessary arguments captured
    def preprocess_func(examples):
        # Tokenize captions using the 'artwork' field
        # Pass the actual tokenizers, not the text encoders
        captions = tokenize_captions((tokenizer, tokenizer_2), examples, text_column=args.caption_column)
        # Preprocess images using the absolute path
        processed_images = preprocess_train(examples, dataset_abs_path, image_transforms, args.image_column)

        # Combine results
        output = {
            "pixel_values": processed_images["pixel_values"],
            "input_ids": captions["input_ids"],
            "attention_mask": captions["attention_mask"],
        }
        # Add text_encoder_2 inputs if they exist
        if 'input_ids_2' in captions:
             output['input_ids_2'] = captions['input_ids_2']
             output['attention_mask_2'] = captions['attention_mask_2']
        return output

    with accelerator.main_process_first():
        # Apply preprocessing to train dataset
        logger.info("Preprocessing training data...")
        train_dataset = train_dataset.map(preprocess_func, batched=True, num_proc=args.preprocessing_num_workers, remove_columns=train_dataset.column_names)
        # Apply preprocessing to val dataset if it exists
        if val_dataset:
            logger.info("Preprocessing validation data...")
            val_dataset = val_dataset.map(preprocess_func, batched=True, num_proc=args.preprocessing_num_workers, remove_columns=val_dataset.column_names)

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
            with accelerator.accumulate(transformer): # Use transformer here
                # Convert images to latent space
                with torch.no_grad(): # VAE encoding should not require gradients
                    # Ensure pixel_values are on the correct device AND dtype
                    pixel_values_device = batch["pixel_values"].to(device=accelerator.device, dtype=torch.float32) # Cast to float32
                    latents = vae.encode(pixel_values_device).latent_dist.sample()
                logger.debug(f"Initial VAE latents shape: {latents.shape}") # Log shape immediately after VAE

                latents = latents * vae.config.scaling_factor
                latents = latents.to(accelerator.device) # Ensure latents are on the correct device

                # Apply projection layer if defined
                logger.debug(f"Shape before projection: {latents.shape}, Target input channels: {transformer_in_channels_actual}")
                if vae_to_transformer_projection is not None:
                    b, c, h, w = latents.shape
                    # Validate the actual channel dimension before projection
                    if c != vae_latent_channels_actual:
                        logger.error(f"PANIC: Latent channels {c} != expected {vae_latent_channels_actual} before projection!")
                        # Handle error appropriately - maybe raise exception
                        raise ValueError(f"Unexpected latent channel dimension: {c}")
                        
                    latents_reshaped = latents.permute(0, 2, 3, 1).reshape(b * h * w, c) # Reshape for Linear layer (B*H*W, C_in=16)
                    projected_latents_reshaped = vae_to_transformer_projection(latents_reshaped) # Apply projection 16->64
                    latents = projected_latents_reshaped.reshape(b, h, w, transformer_in_channels_actual).permute(0, 3, 1, 2) # Reshape back (B, C_out=64, H, W)
                    latents = latents.to(accelerator.device) # Ensure projected latents are on device
                    logger.debug(f"Projected latents shape: {latents.shape}")
                    
                # Reshape latents for transformer: (B, C, H, W) -> (B, H*W, C)
                bsz, channels, height, width = latents.shape
                latents_reshaped = latents.permute(0, 2, 3, 1).reshape(bsz, height * width, channels)
                logger.info(f"Shape AFTER reshape (Input to transformer) - latents_reshaped: {latents_reshaped.shape}")

                # Encode text prompts using the two text encoders
                with torch.no_grad(): # Text encoding should not require gradients
                    prompt_embeds_outputs = text_encoder(
                        batch["input_ids"],
                        output_hidden_states=True,
                    )
                    prompt_embeds = prompt_embeds_outputs.last_hidden_state # Use penultimate layer as recommended
                    clip_pooled = prompt_embeds_outputs.pooler_output # CLIP pooled embeddings

                    prompt_embeds_2_outputs = text_encoder_2(
                        batch["input_ids_2"],
                        output_hidden_states=True,
                    )
                    prompt_embeds_2 = prompt_embeds_2_outputs.last_hidden_state # T5 sequence embeddings

                # Sample noise that we'll use as the target
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Log shapes before transformer call
                logger.debug(f"Shape BEFORE transformer call - latents: {latents.shape}")
                logger.info(f"Shape BEFORE transformer call - timesteps: {timesteps.shape}")
                logger.info(f"Shape BEFORE transformer call - prompt_embeds_2 (T5): {prompt_embeds_2.shape}")
                logger.info(f"Shape BEFORE transformer call - clip_pooled (CLIP): {clip_pooled.shape}")

                # Predict the noise residual using the transformer model
                model_pred = transformer(
                    hidden_states=latents_reshaped.to(accelerator.device), # Pass reshaped latents
                    timestep=timesteps.to(accelerator.device), # Explicitly move timesteps
                    encoder_hidden_states=prompt_embeds_2.to(accelerator.device), # T5 sequence embeddings
                    pooled_projections=clip_pooled.to(accelerator.device), # CLIP pooled embeddings
                ).sample

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
                    # accelerator.clip_grad_norm_(params_to_optimize, 1.0)
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
                            # Fallback if not a PeftModel (shouldn't happen with LoRA but good practice)
                            logger.warning("Attempting to save LoRA checkpoint, but model is not a PeftModel.")
                            # Optionally save the full state dict if needed
                            # torch.save(unwrapped_transformer.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
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
                    pixel_values_device = val_batch["pixel_values"].to(accelerator.device, dtype=torch.float32) # Cast to float32
                    latents = vae.encode(pixel_values_device).latent_dist.sample()
                    logger.debug(f"Validation: Initial VAE latents shape: {latents.shape}") # Log shape

                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(accelerator.device) # Ensure latents are on the correct device

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
                        latents = latents.to(accelerator.device) # Ensure projected latents are on device
                        logger.debug(f"Validation: Projected latents shape: {latents.shape}")
                        
                    # Reshape latents for transformer: (B, C, H, W) -> (B, H*W, C)
                    bsz_val, channels_val, height_val, width_val = latents.shape
                    latents_reshaped_val = latents.permute(0, 2, 3, 1).reshape(bsz_val, height_val * width_val, channels_val)
                    logger.debug(f"Validation shape after reshape: {latents_reshaped_val.shape}")

                    # Encode prompts for validation batch
                    with torch.no_grad():
                        # CLIP Embeddings
                        prompt_embeds_outputs = text_encoder(val_batch["input_ids"].to(accelerator.device), output_hidden_states=True)
                        prompt_embeds = prompt_embeds_outputs.last_hidden_state
                        clip_pooled = prompt_embeds_outputs.pooler_output

                        # T5 Embeddings
                        prompt_embeds_2_outputs = text_encoder_2(val_batch["input_ids_2"].to(accelerator.device), output_hidden_states=True)
                        prompt_embeds_2 = prompt_embeds_2_outputs.last_hidden_state

                    # Sample noise and timesteps for validation
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Predict noise using the model
                    model_pred_val = transformer(
                        hidden_states=latents_reshaped_val.to(accelerator.device),
                        timestep=timesteps.to(accelerator.device),
                        encoder_hidden_states=prompt_embeds_2.to(accelerator.device), # T5 sequence embeddings
                        pooled_projections=clip_pooled.to(accelerator.device), # CLIP pooled embeddings
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
                 torch.save(final_model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
        logger.info(f"Saved final LoRA weights (or full model) to {args.output_dir}")

    accelerator.end_training()
    end_time = time.time()
    total_duration = end_time - training_start_time
    logger.info(f"Training finished in {total_duration:.2f} seconds.")

if __name__ == "__main__":
    # Configuration loading and argument parsing are now handled within parse_args
    args = parse_args()
    
    # Defaults were already handled inside parse_args
    # args.validation_batch_size = getattr(args, 'validation_batch_size', args.batch_size)
    # args.dataloader_num_workers = getattr(args, 'dataloader_num_workers', 4)
    # args.preprocessing_num_workers = getattr(args, 'preprocessing_num_workers', 1)

    main(args)
