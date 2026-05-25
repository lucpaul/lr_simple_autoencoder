# LR UNet Autoencoder (PyTorch + uv)

This project trains a simple image-to-image autoencoder and can be used for inference.
The model is a small 3-level UNet with stride-2 convolutions for downsampling and transposed convolutions for upsampling. It uses no pooling layers.

## Dataset Defaults

The default config uses placeholder paths only.

- Train: `/path/to/your/train_data`
- Validation/Test: set your local validation directory or keep `null` to split from train data

You can change these in `configs/train.yaml`.

## 0) Install uv (if not installed yet)

Pick one method from the official installer.

### Linux / macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your shell (or run the shell init command shown by the installer), and verify:

```bash
uv --version
```

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then open a new terminal and verify:

```powershell
uv --version
```

## 1) Environment Setup with uv

From the project root:

```bash
# Create and activate a virtual environment
uv venv --python 3.13
source .venv/bin/activate

# Install package and dependencies
uv pip install -e .

# Optional: install notebook extras
uv pip install -e .[notebook]
```

You can also skip activation and run commands as `uv run ...`.

## 2) Training

Edit `configs/train.yaml` if needed, then train:

```bash
uv run ae-train --config configs/train.yaml
```

Artifacts are written to `outputs/simple_lr_autoencoder/` by default:

- `checkpoints/best.pt` (lowest validation loss)
- `checkpoints/last.pt`
- `metrics.csv` (epoch-wise train/val loss)

## 3) Inference (GPU or CPU)

Run inference from a checkpoint on one image or a directory.

### Config-driven inference

You can store inference arguments in [configs/infer.yaml](/p/scratch/hai_earth_04/lucas/lr-unet-autoencoder/configs/infer.yaml) and run either entry point with just a config path:

```bash
uv run ae-infer --config configs/infer.yaml
uv run ./src/lr_autoencoder/infer.py --config configs/infer.yaml
```

CLI flags still override values from the config file, for example:

```bash
uv run ae-infer --config configs/infer.yaml --device cuda --tile-stride 128
```

### Single file

```bash
uv run ae-infer \
  --checkpoint outputs/simple_lr_autoencoder/checkpoints/best.pt \
  --input /path/to/image.tif \
  --output outputs/inference \
  --device cpu
```

### Directory of TIFFs

```bash
uv run ae-infer \
  --checkpoint outputs/simple_lr_autoencoder/checkpoints/best.pt \
  --input /path/to/directory/with/tifs \
  --glob "*.tif" \
  --output outputs/inference \
  --device cpu
```

`--device cpu` guarantees CPU-only inference. If omitted, it uses CUDA when available.

## 4) Jupyter Notebook Inference

Install notebook extras and launch Jupyter:

```bash
uv pip install -e .[notebook]
uv run jupyter lab
```

Open `notebooks/inference_example.ipynb`.

The notebook shows how to:

- load `checkpoints/best.pt`
- run inference on a TIFF image
- visualize input vs reconstruction

## Notes

- Input and target are the same LR image (pure autoencoder training).
- Model checkpoints are saved with enough metadata to rebuild the model architecture for CPU or GPU inference.