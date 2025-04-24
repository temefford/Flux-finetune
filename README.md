# FLUX.1-schnell LoRA Fine-Tuning Benchmark

This project fine-tunes the `black-forest-labs/FLUX.1-schnell` model using LoRA to benchmark performance, latency, and cost across different cloud environments (e.g., Runpod, Rackspace, AWS, GCP).

**Note:** This setup uses FLUX.1-schnell. The original benchmark document mentioned `openai/clip-vit-base-patch32`, which would require significant code changes.

## Benchmark Goals

- Performance: Measure validation loss during training. *CLIP score requires `scripts/calc_clip.py` (not included).*
- Latency: Measure training throughput (images/sec) and average step time (ms). *Inference latency (p50/p90/p99) requires `scripts/calc_latency.py` (not included).*
- Cost: Calculate total GPU-hours from training logs and multiply by the provider's hourly rate.

## Setup on Target Environment (e.g., Runpod)

1. Clone Repository:

    ```bash
    # Example: Cloning into /workspace/finetuning on Runpod
    cd /workspace
    git clone <your-repo-url> finetuning
    cd finetuning
    ```

2. Prepare Data:
    - Ensure your dataset (images and `metadata.json`) is placed at the absolute path specified in the config (e.g., `/workspace/art`).
    - Expected structure:

        ```plain
        /workspace/art/
        ├── imgs/
        │   ├── image1.jpg
        │   └── ...
        └── metadata.json
        ```

3. Install Dependencies:
    - **PyTorch:** Install the correct version for your environment's CUDA. Check CUDA with `nvidia-smi` and find install commands at [pytorch.org](https://pytorch.org/get-started/locally/).

        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```

    - **Other Requirements:**

        ```bash
        pip install -r requirements.txt
        ```

    - **Optional (Performance):**

        ```bash
        pip install xformers
        ```

## Configuration

- Review and adjust `configs/ft_config.yaml`.
- Key parameters:
  - `model_id`: Should be `black-forest-labs/FLUX.1-schnell`.
  - `dataset_path`: Absolute path to data (e.g., `/workspace/art`).
  - `device`: `cuda` for NVIDIA.
  - `mixed_precision`: `bf16` recommended for H100/H200/A100.
  - `batch_size`, `gradient_accumulation_steps`, `learning_rate`, `epochs`, `lora_rank`.

## Running the Benchmark

1. Run Fine-Tuning (`ft_test.py`):

    ```bash
    # Navigate to the repo root (/workspace/finetuning)
    python scripts/ft_test.py --config configs/ft_config.yaml
    ```

    - This script logs per-epoch validation loss and training throughput (images/sec).
    - It saves the best checkpoint (lowest validation loss) to the `output_dir` specified in the config.
    - It logs the total training duration at the end.

2. Measure CLIP Score (Requires `calc_clip.py` - Not Included):

    ```bash
    # Placeholder command
    # python scripts/calc_clip.py \
    #   --model_path <output_dir>/best_checkpoint \
    #   --data_dir <dataset_path>/val # Assuming validation split data exists
    ```

3. Measure Inference Latency (Requires `calc_latency.py` - Not Included):

    ```bash
    # Placeholder command
    # python scripts/calc_latency.py \
    #   --model_path <output_dir>/best_checkpoint \
    #   --prompts_file <path_to_prompts.txt> \
    #   --runs 100
    ```

4. Calculate Cost:
    - Find the `total_duration` (in seconds) in the `ft_test.py` log output.
    - Convert duration to hours: `total_hours = total_duration / 3600`.
    - Multiply by the cloud provider's $/GPU-hour rate: `total_cost = total_hours * hourly_rate`.

## Metrics Captured by `ft_test.py`

- **Performance:** Validation loss (per epoch).
- **Latency:** Training throughput (images/sec), Average training step time (ms).
- **Cost:** Total training time (seconds) - requires manual multiplication by hourly rate.

## Comparing Environments

Run the *exact same setup* (code, config, data) on Rackspace, AWS, and GCP instances to get comparable results for performance, latency (training), and cost.
