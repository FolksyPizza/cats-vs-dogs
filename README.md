# Cats vs Dogs

Concise, in‑depth guide to run multi‑GPU training, understand how it works, and what’s optimized.

## Quick Start
```bash
pip install -r requirements.txt
./run_train.sh
```

## Dataset Layout
Expect `ImageFolder` style directories:
```
/ssd/datasets/train/
  cats/...
  dogs/...
/ssd/datasets/val/
  cats/...
  dogs/...
```
Override with `--train-dir` and `--val-dir` if needed.

## What Runs
`run_train.sh`:
- Creates a default Accelerate config if missing.
- Launches 2 GPU processes (`--multi_gpu --num_processes 2`).
- Passes all extra arguments through to `train.py`.

`train.py`:
- Builds transforms, datasets, loaders.
- Creates the model with `timm`.
- Trains with AMP + mixed precision.
- Logs to TensorBoard.
- Saves `checkpoints/best.pt` and `checkpoints/last.pt`.

## Core Optimizations
Compute‑focused:
- **Mixed precision (FP16/BF16)** via Accelerate to boost throughput on 3090s.
- **TF32** enabled for matmuls and cuDNN.
- **Channels‑last memory format** for better convolution performance.
- **Pinned memory + persistent workers** to keep GPUs fed.
- **OneCycleLR** for fast convergence at higher LR.
- **AdamW** with decoupled weight decay for stable training.

Generalization‑focused:
- **AutoAugment + RandomResizedCrop + RandomErasing**.
- **Mixup/Cutmix** for smoother decision boundaries.
- **Label smoothing** to reduce overconfidence.
- **Drop‑path** (stochastic depth) for better regularization on EfficientNetV2.

## Key Defaults (Tuned for 2×3090)
```
--model tf_efficientnetv2_s
--img-size 256
--batch-size 128
--epochs 30
--lr 1e-3
--mixup 0.4
--cutmix 0.1
--label-smoothing 0.1
--drop-path 0.1
--mixed-precision fp16
```

## Customize Training
Examples:
```bash
# Smaller image size for speed
./run_train.sh --img-size 224

# Larger effective batch via grad accumulation
./run_train.sh --batch-size 96 --grad-accum-steps 2

# Switch to BF16 if stable
./run_train.sh --mixed-precision bf16

# Enable torch.compile (PyTorch 2.x)
./run_train.sh --compile
```

## Logs & Monitoring
TensorBoard logs in `runs/`:
```bash
tensorboard --logdir runs
```

## Checkpoints
- Best model: `checkpoints/best.pt`
- Final model: `checkpoints/last.pt`

## Notes
- If the dataset is large, increase `--num-workers`.
- If you hit VRAM limits, reduce `--batch-size` or `--img-size`.
- For single‑GPU, pass `--num_processes 1` in `run_train.sh`.
