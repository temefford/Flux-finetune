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
import transformers
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers import FluxPipeline
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig, PeftModel
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer

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
        logger.error(f"Configuration file not found at: {cmd_args.config}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration file {cmd_args.config}: {e}")
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
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Models and Tokenizers ---
    # FLUX uses multiple components, specific loading might differ slightly
    # Based on common patterns, adjust if FLUX requires specific handling
    try:
        pipe = FluxPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32)
        tokenizer = (pipe.tokenizer, pipe.tokenizer_2)
        text_encoder = (pipe.text_encoder, pipe.text_encoder_2)
        vae = pipe.vae
        unet = pipe.transformer # FLUX uses 'transformer' instead of 'unet'
        scheduler = pipe.scheduler # Use the scheduler from the pipeline
        logger.info("Loaded FLUX model components.")
    except Exception as e:
        logger.error(f"Failed to load FLUX model components: {e}")
        logger.error("Ensure you have 'transformers', 'diffusers', 'torch' installed and are logged in to Hugging Face Hub if needed.")
        logger.error("Also check the model identifier: {args.model_id}")
        return

    # Freeze VAE and text_encoder
    vae.requires_grad_(False)
    text_encoder[0].requires_grad_(False)
    text_encoder[1].requires_grad_(False)
    unet.requires_grad_(False) # Start with UNet frozen, LoRA will unfreeze target modules

    # --- Add LoRA to UNet (Transformer in FLUX) ---
    if args.peft_method == "LoRA":
        logger.info("Adding LoRA layers to the Transformer (UNet equivalent).")
        unet_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank, # Often set equal to rank
            # Target the Q, K, V projections based on printed names
            target_modules=["to_q", "to_k", "to_v"], # Corrected target modules
            lora_dropout=0.1, # Optional dropout
            bias="none",
        )
        unet.add_adapter(unet_lora_config)
        # Ensure LoRA layers are float32 for stability if using mixed precision
        if args.mixed_precision == "fp16":
             for name, param in unet.named_parameters():
                if "lora_" in name:
                    param.data = param.data.to(torch.float32)
        logger.info(f"Added LoRA with rank {args.lora_rank} to {unet_lora_config.target_modules}")

    else:
        logger.warning(f"PEFT method '{args.peft_method}' not implemented for this script. Training full model.")
        unet.requires_grad_(True) # Train full UNet if not LoRA

    # --- Optimizer Setup ---
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
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
        captions = tokenize_captions(tokenizer, examples, text_column=args.caption_column) # Use caption_column from args
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
        unet, optimizer, train_dataloader, lr_scheduler, val_dataloader = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler, val_dataloader
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # Move vae and text_encoder to device
    vae.to(accelerator.device)
    text_encoder[0].to(accelerator.device)
    text_encoder[1].to(accelerator.device)
    if args.mixed_precision == "fp16":
        vae.to(dtype=torch.float16)
        # Text encoder precision depends on the model, often kept in fp32 or needs specific handling
        # text_encoder.to(dtype=torch.float16)

    # --- Training Loop ---
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.epochs * num_update_steps_per_epoch
    if args.max_train_steps is not None and args.max_train_steps > 0:
        max_train_steps = min(max_train_steps, args.max_train_steps)
        args.epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

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
        unet.train()
        train_loss = 0.0
        steps_in_epoch = 0 # Track optimization steps within the epoch
        images_processed_epoch = 0 # Track images processed in this epoch
        epoch_start_time = time.time() # Record time at the start of the epoch

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    # Check if VAE is a compiled module
                    if is_compiled_module(vae):
                        latents = vae.encode(batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)).latent_dist.sample()
                    else:
                         latents = vae.encode(batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                with torch.no_grad():
                    encoder_hidden_states = text_encoder[0](batch["input_ids"].to(accelerator.device))[0]
                    encoder_hidden_states_2 = text_encoder[1](batch["input_ids_2"].to(accelerator.device))[0]
                    # Handle potential pooled output requirement for FLUX if needed
                    # text_embeds = text_encoder(batch["input_ids"].to(accelerator.device)).pooled_output

                # Combine embeddings if needed or pass separately
                # Assuming separate args for now, adjust based on FLUX model signature
                prompt_embeds = encoder_hidden_states # TBC if needs combining
                prompt_embeds_2 = encoder_hidden_states_2

                # Predict the noise residual
                # Pass both encoder hidden states to the transformer
                model_pred = unet(
                    noisy_latents,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_2=prompt_embeds_2, # Pass second embedding
                    attention_mask=batch["attention_mask"], # May need attention_mask_2 as well?
                    timestep=timesteps
                ).sample

                # Get the target for loss depending on the prediction type
                if scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif scheduler.config.prediction_type == "v_prediction":
                    target = scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")

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
                        # Save LoRA layers specifically
                        unet_lora_state_dict = get_peft_model_state_dict(accelerator.unwrap_model(unet))
                        LoraLoaderMixin.save_lora_weights(save_path, unet_lora_state_dict=unet_lora_state_dict)
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
            unet.eval() # Set model to evaluation mode
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad(): # Disable gradient calculations for validation
                for val_step, val_batch in enumerate(tqdm(val_dataloader, desc="Validation", disable=not accelerator.is_local_main_process)):
                    # --- Validation Step Logic (mirrors training step for loss calculation) --- 
                    # Convert images to latent space
                    if is_compiled_module(vae):
                        latents = vae.encode(val_batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)).latent_dist.sample()
                    else:
                        latents = vae.encode(val_batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()
                    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                    # Get text embeddings
                    encoder_hidden_states = text_encoder[0](val_batch["input_ids"].to(accelerator.device))[0]
                    encoder_hidden_states_2 = text_encoder[1](val_batch["input_ids_2"].to(accelerator.device))[0]
                    prompt_embeds = encoder_hidden_states
                    prompt_embeds_2 = encoder_hidden_states_2

                    # Predict the noise residual
                    model_pred = unet(
                        noisy_latents,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_2=prompt_embeds_2,
                        attention_mask=val_batch["attention_mask"],
                        timestep=timesteps
                    ).sample

                    # Get the target for loss
                    if scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif scheduler.config.prediction_type == "v_prediction":
                        target = scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")

                    batch_val_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    # --- End Validation Step Logic --- 

                    # Gather loss across processes
                    # Use validation batch size for gathering
                    avg_batch_val_loss = accelerator.gather(batch_val_loss.repeat(args.validation_batch_size)).mean()
                    val_loss += avg_batch_val_loss.item()
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
                    
                    unwrapped_unet = accelerator.unwrap_model(unet)
                    # Save using appropriate method (PEFT LoRA or standard HF)
                    if isinstance(unwrapped_unet, PeftModel):
                         # Save LoRA weights specifically if using PEFT library
                         unwrapped_unet.save_pretrained(best_checkpoint_dir)
                    elif hasattr(unwrapped_unet, 'save_pretrained'):
                         # Standard Hugging Face save_pretrained for non-PEFT models or if LoRA is merged
                         unwrapped_unet.save_pretrained(best_checkpoint_dir)
                    else:
                         # Fallback or add specific logic if using a custom model structure
                         torch.save(unwrapped_unet.state_dict(), os.path.join(best_checkpoint_dir, "pytorch_model.bin"))
                    logger.info(f"New best validation loss: {best_val_loss:.4f}. Saved checkpoint to {best_checkpoint_dir}")
            # Set model back to train mode after validation
            unet.train()
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
        unet_final = accelerator.unwrap_model(unet)
        unet_lora_state_dict = get_peft_model_state_dict(unet_final)
        LoraLoaderMixin.save_lora_weights(
            args.output_dir,
            unet_lora_state_dict=unet_lora_state_dict
        )
        logger.info(f"Saved final LoRA weights to {args.output_dir}")

        # Save config file
        with open(os.path.join(args.output_dir, 'final_config.yaml'), 'w') as f:
            yaml.dump(vars(args), f)

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
