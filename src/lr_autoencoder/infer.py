from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
import torch

from lr_autoencoder.config import InferenceConfig, load_inference_config
from lr_autoencoder.data import (
    _center_crop,
    _per_image_minmax,
    _to_chw_float32,
    get_tiff_chw_shape,
    make_tiled_inference_loader,
)
from lr_autoencoder.model import SimpleUNetAutoencoder


def _build_parser(defaults: InferenceConfig | None = None) -> argparse.ArgumentParser:
    defaults = defaults or InferenceConfig()

    parser = argparse.ArgumentParser(description="Run inference with a trained UNet autoencoder checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML inference config")
    parser.add_argument("--checkpoint", type=str, default=defaults.checkpoint, help="Path to best.pt or last.pt")
    parser.add_argument("--input", type=str, default=defaults.input, help="Input TIFF file or directory")
    parser.add_argument("--output", type=str, default=defaults.output, help="Output directory")
    parser.add_argument("--glob", type=str, default=defaults.glob, help="Glob pattern when --input is a directory")
    parser.add_argument(
        "--inference-mode",
        type=str,
        default=defaults.inference_mode,
        choices=["single", "tiled"],
        help="Inference mode: 'single' loads the full image at once; 'tiled' uses the memory-safe tiling dataloader.",
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size, help="Inference batch size for tiled inference")
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers, help="DataLoader workers for tiled inference")
    parser.add_argument("--tile-size", type=int, default=defaults.tile_size, help="Tile size for large-image tiled inference")
    parser.add_argument(
        "--tile-stride",
        type=int,
        default=defaults.tile_stride,
        help="Tile stride for large-image tiled inference",
    )
    parser.add_argument(
        "--save-prepared-input",
        type=str,
        default=defaults.save_prepared_input,
        help="Optional path to save full prepared input array used for inference.",
    )
    parser.add_argument(
        "--prepared-input-dtype",
        type=str,
        default=defaults.prepared_input_dtype,
        choices=["float32", "float64", "uint16", "uint8"],
        help="Dtype for saved prepared input array.",
    )
    parser.add_argument(
        "--prepared-input-raw",
        action="store_true",
        default=defaults.prepared_input_raw,
        help="Save prepared input without normalization (raw converted dtype).",
    )
    parser.add_argument(
        "--save-residual",
        action="store_true",
        default=defaults.save_residual,
        help="Also save residual = prediction - prepared_input (float32).",
    )
    parser.add_argument(
        "--stitch-mode",
        type=str,
        default=defaults.stitch_mode,
        choices=["average", "crop", "valid"],
        help="Tile stitching mode. 'valid' uses model loaded with conv padding=0.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=defaults.device,
        choices=["cpu", "cuda"],
        help="Inference device. Defaults to cuda if available, else cpu.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args()

    defaults = load_inference_config(config_args.config) if config_args.config else InferenceConfig()
    parser = _build_parser(defaults)
    args = parser.parse_args()

    missing = [name for name in ("checkpoint", "input", "output") if getattr(args, name) is None]
    if missing:
        parser.error(f"Missing required inference arguments: {', '.join('--' + name for name in missing)}")

    return args


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    conv_padding: int = 1,
) -> tuple[SimpleUNetAutoencoder, dict]:
    device = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_cfg = ckpt["model_config"]
    model = SimpleUNetAutoencoder(
        in_channels=model_cfg["in_channels"],
        base_channels=model_cfg["base_channels"],
        conv_padding=conv_padding,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def _compute_output_tile_size(model: torch.nn.Module, in_channels: int, tile_size: int, device: str | torch.device) -> int:
    device = torch.device(device)
    with torch.no_grad():
        x = torch.zeros((1, in_channels, tile_size, tile_size), device=device, dtype=torch.float32)
        y = model(x)
    if y.shape[-2] != y.shape[-1]:
        raise ValueError(f"Expected square tile output, got {tuple(y.shape[-2:])}")
    return int(y.shape[-1])


def _stitch_average(
    preds: list[np.ndarray],
    tops: list[int],
    lefts: list[int],
    out_h: int,
    out_w: int,
    out_patch: int,
) -> np.ndarray:
    c = preds[0].shape[0]
    acc = np.zeros((c, out_h, out_w), dtype=np.float32)
    cnt = np.zeros((1, out_h, out_w), dtype=np.float32)

    for pred, top, left in zip(preds, tops, lefts):
        h_end = min(top + out_patch, out_h)
        w_end = min(left + out_patch, out_w)
        ph = h_end - top
        pw = w_end - left
        acc[:, top:h_end, left:w_end] += pred[:, :ph, :pw]
        cnt[:, top:h_end, left:w_end] += 1.0

    return acc / np.clip(cnt, 1e-6, None)


def _stitch_crop(
    preds: list[np.ndarray],
    tops: list[int],
    lefts: list[int],
    out_h: int,
    out_w: int,
    out_patch: int,
    stride: int,
) -> np.ndarray:
    if out_patch < stride:
        raise ValueError(f"For crop stitching, out_patch ({out_patch}) must be >= stride ({stride})")

    c = preds[0].shape[0]
    out = np.zeros((c, out_h, out_w), dtype=np.float32)
    ny = (out_h - out_patch) // stride + 1 if out_h >= out_patch else 1
    nx = (out_w - out_patch) // stride + 1 if out_w >= out_patch else 1

    margin = out_patch - stride
    crop_before = margin // 2
    crop_after = margin - crop_before

    for pred, top, left in zip(preds, tops, lefts):
        ty = top // stride
        tx = left // stride

        ct = 0 if ty == 0 else crop_before
        cl = 0 if tx == 0 else crop_before
        cb = 0 if ty == ny - 1 else crop_after
        cr = 0 if tx == nx - 1 else crop_after

        tile = pred[:, ct : out_patch - cb, cl : out_patch - cr]
        dst_top = top + ct
        dst_left = left + cl
        dst_bottom = min(dst_top + tile.shape[1], out_h)
        dst_right = min(dst_left + tile.shape[2], out_w)

        out[:, dst_top:dst_bottom, dst_left:dst_right] = tile[:, : dst_bottom - dst_top, : dst_right - dst_left]

    return out


def infer_image_tiled(
    model: torch.nn.Module,
    image_path: str | Path,
    device: str | torch.device,
    tile_size: int,
    tile_stride: int,
    batch_size: int,
    num_workers: int,
    stitch_mode: str,
    save_prepared_input_path: str | Path | None = None,
    save_prepared_input_dtype: str = "float32",
    save_prepared_input_normalized: bool = True,
    return_prepared_input: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    loader, ds = make_tiled_inference_loader(
        image_path=image_path,
        patch_size=tile_size,
        stride=tile_stride,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=True,
        save_prepared_input_path=save_prepared_input_path,
        save_prepared_input_dtype=save_prepared_input_dtype,
        save_prepared_input_normalized=save_prepared_input_normalized,
    )

    preds: list[np.ndarray] = []
    tops: list[int] = []
    lefts: list[int] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            y = model(x).detach().cpu().numpy().astype(np.float32)
            preds.extend([y[i] for i in range(y.shape[0])])
            tops.extend(batch["top"].tolist())
            lefts.extend(batch["left"].tolist())

    out_h = ds.spec.padded_h
    out_w = ds.spec.padded_w
    out_patch = int(preds[0].shape[-1])

    if stitch_mode == "average":
        recon = _stitch_average(preds, tops, lefts, out_h=out_h, out_w=out_w, out_patch=out_patch)
    elif stitch_mode == "crop":
        recon = _stitch_crop(
            preds,
            tops,
            lefts,
            out_h=out_h,
            out_w=out_w,
            out_patch=out_patch,
            stride=tile_stride,
        )
    else:
        raise ValueError(f"Unsupported stitch mode: {stitch_mode}")

    recon = np.transpose(recon[:, : ds.spec.orig_h, : ds.spec.orig_w], (1, 2, 0)).astype(np.float32)
    prepared = None
    if return_prepared_input:
        prepared_chw = ds.get_prepared_input_array(normalized=save_prepared_input_normalized, dtype=np.float32)
        prepared = np.transpose(prepared_chw, (1, 2, 0)).astype(np.float32)
    return recon, prepared


def infer_image_tiled_valid_padding(
    model: torch.nn.Module,
    image_path: str | Path,
    device: str | torch.device,
    tile_size: int,
    batch_size: int,
    num_workers: int,
    save_prepared_input_path: str | Path | None = None,
    save_prepared_input_dtype: str = "float32",
    save_prepared_input_normalized: bool = True,
    return_prepared_input: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    in_channels, h, w = get_tiff_chw_shape(image_path)

    out_patch = _compute_output_tile_size(model, in_channels=in_channels, tile_size=tile_size, device=device)
    if out_patch <= 0 or out_patch > tile_size:
        raise ValueError(f"Unexpected output patch size {out_patch} for tile_size={tile_size}")

    border = (tile_size - out_patch) // 2
    stride = out_patch

    loader, ds = make_tiled_inference_loader(
        image_path=image_path,
        patch_size=tile_size,
        stride=stride,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=True,
        pad_top=border,
        pad_bottom=border,
        pad_left=border,
        pad_right=border,
        save_prepared_input_path=save_prepared_input_path,
        save_prepared_input_dtype=save_prepared_input_dtype,
        save_prepared_input_normalized=save_prepared_input_normalized,
    )

    full_out_h = ds.spec.padded_h - 2 * border
    full_out_w = ds.spec.padded_w - 2 * border
    c = in_channels
    out = np.zeros((c, full_out_h, full_out_w), dtype=np.float32)

    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            y = model(x).detach().cpu().numpy().astype(np.float32)
            tops = batch["top"].tolist()
            lefts = batch["left"].tolist()
            for i in range(y.shape[0]):
                top_out = tops[i]
                left_out = lefts[i]
                out[:, top_out : top_out + out_patch, left_out : left_out + out_patch] = y[i]

    out = out[:, :h, :w]
    recon = np.transpose(out, (1, 2, 0)).astype(np.float32)
    prepared = None
    if return_prepared_input:
        prepared_chw = ds.get_prepared_input_array(normalized=save_prepared_input_normalized, dtype=np.float32)
        prepared = np.transpose(prepared_chw, (1, 2, 0)).astype(np.float32)
    return recon, prepared


def infer_image(model: torch.nn.Module, image_path: str | Path, image_size: int, device: str | torch.device) -> np.ndarray:
    arr = tifffile.imread(image_path)
    x = _to_chw_float32(arr)
    x = _center_crop(x, image_size)
    x = _per_image_minmax(x)
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        y = model(x)

    y = y.squeeze(0).detach().cpu().numpy()
    y = np.transpose(y, (1, 2, 0))
    return y.astype(np.float32)


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    conv_padding = 0 if args.stitch_mode == "valid" else 1
    model, ckpt = load_model_from_checkpoint(args.checkpoint, device=device, conv_padding=conv_padding)
    image_size = int(ckpt.get("data_config", {}).get("image_size", 192))

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        paths = [input_path]
    else:
        paths = sorted(input_path.glob(args.glob))
        if not paths and args.glob == "*.tif":
            paths = sorted(input_path.glob("*.tiff"))

    if not paths:
        raise FileNotFoundError(f"No input files found from {input_path} with pattern {args.glob}")

    for path in paths:
        prepared = None
        prepared_save_path = None
        if args.save_prepared_input:
            base = Path(args.save_prepared_input)
            if base.suffix.lower() in {".tif", ".tiff"}:
                prepared_save_path = base.parent / f"{base.stem}_{path.stem}{base.suffix}"
            else:
                prepared_save_path = base / f"{path.stem}_prepared_input.tif"

        if args.inference_mode == "single":
            recon = infer_image(model, path, image_size=image_size, device=device)
            if args.save_residual:
                arr = tifffile.imread(path)
                x = _to_chw_float32(arr)
                x = _center_crop(x, image_size)
                x = _per_image_minmax(x)
                prepared = np.transpose(x.numpy(), (1, 2, 0)).astype(np.float32)
            if prepared_save_path is not None:
                arr = tifffile.imread(path)
                x = _to_chw_float32(arr)
                if not args.prepared_input_raw:
                    x = _per_image_minmax(x)
                prepared_chw = x.numpy().astype(np.float32)
                prepared_save_path.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(prepared_save_path, prepared_chw.astype(np.dtype(args.prepared_input_dtype), copy=False))
        elif args.stitch_mode == "valid":
            recon, prepared = infer_image_tiled_valid_padding(
                model=model,
                image_path=path,
                device=device,
                tile_size=args.tile_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                save_prepared_input_path=prepared_save_path,
                save_prepared_input_dtype=args.prepared_input_dtype,
                save_prepared_input_normalized=not args.prepared_input_raw,
                return_prepared_input=args.save_residual,
            )
        else:
            recon, prepared = infer_image_tiled(
                model=model,
                image_path=path,
                device=device,
                tile_size=args.tile_size,
                tile_stride=args.tile_stride,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                stitch_mode=args.stitch_mode,
                save_prepared_input_path=prepared_save_path,
                save_prepared_input_dtype=args.prepared_input_dtype,
                save_prepared_input_normalized=not args.prepared_input_raw,
                return_prepared_input=args.save_residual,
            )

        out_path = output_dir / f"{path.stem}_recon.tif"
        tifffile.imwrite(out_path, recon)
        print(f"Wrote {out_path}")

        if prepared_save_path is not None:
            print(f"Wrote {prepared_save_path}")

        if args.save_residual:
            if prepared is None:
                raise RuntimeError("Residual requested but prepared input is unavailable")
            if prepared.shape != recon.shape:
                raise ValueError(f"Residual shape mismatch: recon={recon.shape}, prepared={prepared.shape}")
            residual = (recon - prepared).astype(np.float32)
            residual_path = output_dir / f"{path.stem}_residual.tif"
            tifffile.imwrite(residual_path, residual)
            print(f"Wrote {residual_path}")


if __name__ == "__main__":
    main()