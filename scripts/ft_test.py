# ft_test.py  (clean version – Python 3.10+)

import argparse, logging, math, os, time
from pathlib import Path

import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator, ProjectConfiguration, set_seed
from accelerate.logging import get_logger
from datasets import load_dataset
from diffusers import FluxPipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig, PeftModel
from PIL import Image
import yaml
from tqdm.auto import tqdm

logger = get_logger(__name__, log_level="INFO")

# ------------------------------------------------------------------ #
#                            ARGUMENTS                               #
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ft_config.yaml")
    p.add_argument("--data_dir")
    p.add_argument("--output_dir")
    cmd = p.parse_args()

    with open(cmd.config, "r") as f:
        cfg = yaml.safe_load(f)
    args = argparse.Namespace(**cfg)

    if cmd.data_dir:
        args.dataset_path = cmd.data_dir
    if cmd.output_dir:
        args.output_dir = cmd.output_dir

    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    if not getattr(args, "dataset_path", None):
        raise ValueError("`dataset_path` missing (config or --data_dir)")

    # sensible defaults
    args.validation_batch_size = getattr(args, "validation_batch_size", args.batch_size)
    args.dataloader_num_workers = getattr(args, "dataloader_num_workers", 4)
    args.preprocessing_num_workers = getattr(args, "preprocessing_num_workers", 1)
    return args


# ------------------------------------------------------------------ #
#                    CAPTION TOKENISATION HELPERS                    #
# ------------------------------------------------------------------ #
def tokenize_captions(tokenizers, examples, text_column="caption"):
    clip_tok, t5_tok = tokenizers
    captions = [str(c[0] if isinstance(c, (list, tuple)) else c) for c in examples[text_column]]

    clip = clip_tok(captions, padding="max_length", truncation=True, return_tensors="pt")
    t5   = t5_tok(captions,   padding="max_length", truncation=True, return_tensors="pt")
    return {
        "input_ids":        clip.input_ids,
        "attention_mask":   clip.attention_mask,
        "input_ids_2":      t5.input_ids,
        "attention_mask_2": t5.attention_mask,
    }


def preprocess_train(examples, root, tfms, img_col):
    img_dir = os.path.join(root, "imgs")
    images = [Image.open(os.path.join(img_dir, f"{name}.jpg")).convert("RGB") for name in examples[img_col]]
    examples["pixel_values"] = [tfms(im) for im in images]
    return examples


# ------------------------------------------------------------------ #
#                               MAIN                                 #
# ------------------------------------------------------------------ #
def main(args: argparse.Namespace):
    logging_dir = Path(args.output_dir, "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=ProjectConfiguration(args.output_dir, logging_dir),
        log_with="tensorboard",
    )
    set_seed(args.seed)

    weight_dtype = (
        torch.float16 if args.mixed_precision == "fp16" else
        torch.bfloat16 if args.mixed_precision == "bf16" else
        torch.float32
    )

    # --------------------  LOAD FLUX  ------------------------------ #
    pipe = FluxPipeline.from_pretrained(args.model_id, torch_dtype=weight_dtype)
    vae, transformer = pipe.vae, pipe.transformer
    text_encoder, text_encoder_2 = pipe.text_encoder, pipe.text_encoder_2
    tokenizer, tokenizer_2 = pipe.tokenizer, pipe.tokenizer_2
    noise_sched = pipe.scheduler

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # -----------------  ADD LoRA TO TRANSFORMER  ------------------ #
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    transformer.add_adapter(lora_cfg)

    # ---------------  VAE → TRANSFORMER PROJECTION  --------------- #
    proj = torch.nn.Conv2d(16, 64, kernel_size=1, bias=False).to(accelerator.device, dtype=weight_dtype)

    params = [p for p in transformer.parameters() if p.requires_grad] + list(proj.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(args.learning_rate), betas=(0.9, 0.999), weight_decay=1e-2)

    # ---------------------  DATASET  ------------------------------- #
    root = os.path.abspath(args.dataset_path)
    dataset = load_dataset("json", data_files=os.path.join(root, "metadata.json"), split="train")
    split = dataset.train_test_split(test_size=args.val_split, seed=args.seed) if args.val_split else {"train": dataset}
    train_ds, val_ds = split["train"], split.get("test")

    tfms = transforms.Compose([
        transforms.Resize(args.image_resolution, transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.image_resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def preprocess(ex):
        caps = tokenize_captions((tokenizer, tokenizer_2), ex, args.caption_column)
        imgs = preprocess_train(ex, root, tfms, args.image_column)
        return {**caps, "pixel_values": imgs["pixel_values"]}

    with accelerator.main_process_first():
        train_ds = train_ds.map(preprocess, batched=True, num_proc=args.preprocessing_num_workers,
                                remove_columns=train_ds.column_names)
        if val_ds:
            val_ds   = val_ds.map(preprocess,   batched=True, num_proc=args.preprocessing_num_workers,
                                  remove_columns=val_ds.column_names)

    def collate(batch):
        keys = batch[0].keys()
        out = {k: torch.stack([x[k] for x in batch]) for k in keys}
        return out

    train_dl = DataLoader(train_ds, shuffle=True,  batch_size=args.batch_size,
                          num_workers=args.dataloader_num_workers, collate_fn=collate)
    val_dl   = DataLoader(val_ds, shuffle=False, batch_size=args.validation_batch_size,
                          num_workers=args.dataloader_num_workers, collate_fn=collate) if val_ds else None

    # -------------  LR SCHEDULER after step count ----------------- #
    steps_per_epoch = math.ceil(len(train_dl) / args.gradient_accumulation_steps)
    max_steps = (args.epochs * steps_per_epoch) if args.max_train_steps < 0 else args.max_train_steps
    lr_sched = get_scheduler("cosine", optimizer, 50, max_steps)

    # ---------------  PREPARE FOR DISTRIBUTED  -------------------- #
    to_prep = (transformer, proj, optimizer, train_dl, lr_sched) + ((val_dl,) if val_dl else ())
    prepared = accelerator.prepare(*to_prep)
    transformer, proj, optimizer, train_dl, lr_sched = prepared[:5]
    val_dl = prepared[5] if val_dl else None

    # Send frozen encoders to device
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    # --------------------  TRAINING LOOP  ------------------------- #
    global_step = 0
    progress = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)
    for epoch in range(args.epochs):
        transformer.train()
        for batch in train_dl:
            with accelerator.accumulate(transformer):
                pix = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                lat = vae.encode(pix).latent_dist.sample() * vae.config.scaling_factor
                lat = proj(lat)                                      # (B,64,H,W)
                b, c, h, w = lat.shape
                lat_tok = lat.permute(0, 2, 3, 1).reshape(b, h * w, c)

                noise = torch.randn_like(lat_tok)
                ts = torch.randint(0, noise_sched.config.num_train_timesteps, (b,), device=lat_tok.device).long()

                with torch.no_grad():
                    clip_out = text_encoder(batch["input_ids"].to(accelerator.device), output_hidden_states=True)
                    t5_out   = text_encoder_2(batch["input_ids_2"].to(accelerator.device), output_hidden_states=True)

                pred = transformer(
                    hidden_states      = lat_tok,
                    timestep           = ts,
                    encoder_hidden_states = t5_out.last_hidden_state,
                    pooled_projections    = clip_out.pooler_output,
                ).sample

                loss = F.mse_loss(pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)

                optimizer.step()
                lr_sched.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress.update(1)
                global_step += 1
                if global_step >= max_steps:
                    break
        if global_step >= max_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(transformer).save_pretrained(args.output_dir)
    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())