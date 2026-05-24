from __future__ import annotations

from pathlib import Path
from typing import Sequence
from dataclasses import dataclass

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset


def get_tiff_chw_shape(image_path: str | Path) -> tuple[int, int, int]:
    """Return image shape as (C, H, W) without requiring memmap support."""
    image_path = Path(image_path)
    try:
        raw = tifffile.memmap(image_path)
        shape = raw.shape
    except ValueError:
        # For compressed/non-contiguous TIFFs, use metadata shape from first series.
        with tifffile.TiffFile(image_path) as tif:
            shape = tif.series[0].shape

    if len(shape) == 2:
        h, w = int(shape[0]), int(shape[1])
        return 1, h, w
    if len(shape) == 3:
        if shape[0] <= 8 and shape[1] > 8 and shape[2] > 8:
            return int(shape[0]), int(shape[1]), int(shape[2])
        return int(shape[2]), int(shape[0]), int(shape[1])
    raise ValueError(f"Expected 2D/3D TIFF, got shape {shape}")


def _resolve_numpy_dtype(dtype: str | np.dtype) -> np.dtype:
    return np.dtype(dtype)


def _list_tiff_files(root: Path) -> list[Path]:
    files = sorted(root.glob("*.tif")) + sorted(root.glob("*.tiff"))
    if not files:
        raise FileNotFoundError(f"No .tif/.tiff files found in {root}")
    return files


def _to_chw_float32(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 3:
        if arr.shape[0] <= 8 and arr.shape[1] > 8 and arr.shape[2] > 8:
            pass
        else:
            arr = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError(f"Expected 2D/3D TIFF, got shape {arr.shape}")
    return torch.from_numpy(arr.astype(np.float32))


def _center_crop(t: torch.Tensor, size: int) -> torch.Tensor:
    _, h, w = t.shape
    if h < size or w < size:
        pad_h = max(size - h, 0)
        pad_w = max(size - w, 0)
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h))
        _, h, w = t.shape
    top = (h - size) // 2
    left = (w - size) // 2
    return t[:, top : top + size, left : left + size]


def _per_image_minmax(x: torch.Tensor) -> torch.Tensor:
    x_min = x.amin(dim=(1, 2), keepdim=True)
    x_max = x.amax(dim=(1, 2), keepdim=True)
    return (x - x_min) / (x_max - x_min + 1e-8)


@dataclass
class TileGridSpec:
    orig_h: int
    orig_w: int
    padded_h: int
    padded_w: int
    patch_size: int
    stride: int
    num_tiles_y: int
    num_tiles_x: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int


def _compute_tail_pad(length: int, patch_size: int, stride: int) -> int:
    if length <= patch_size:
        return patch_size - length
    rem = (length - patch_size) % stride
    return 0 if rem == 0 else stride - rem


def make_tile_grid_spec(
    orig_h: int,
    orig_w: int,
    patch_size: int,
    stride: int,
    pad_top: int = 0,
    pad_bottom: int = 0,
    pad_left: int = 0,
    pad_right: int = 0,
) -> TileGridSpec:
    base_h = orig_h + pad_top + pad_bottom
    base_w = orig_w + pad_left + pad_right

    tail_h = _compute_tail_pad(base_h, patch_size, stride)
    tail_w = _compute_tail_pad(base_w, patch_size, stride)

    padded_h = base_h + tail_h
    padded_w = base_w + tail_w
    final_pad_bottom = pad_bottom + tail_h
    final_pad_right = pad_right + tail_w

    num_tiles_y = ((padded_h - patch_size) // stride) + 1
    num_tiles_x = ((padded_w - patch_size) // stride) + 1

    return TileGridSpec(
        orig_h=orig_h,
        orig_w=orig_w,
        padded_h=padded_h,
        padded_w=padded_w,
        patch_size=patch_size,
        stride=stride,
        num_tiles_y=num_tiles_y,
        num_tiles_x=num_tiles_x,
        pad_top=pad_top,
        pad_bottom=final_pad_bottom,
        pad_left=pad_left,
        pad_right=final_pad_right,
    )


class TiffTiledInferenceDataset(Dataset):
    """Memory-safe tiled reader for large TIFF inference."""

    def __init__(
        self,
        image_path: str | Path,
        patch_size: int,
        stride: int,
        normalize: bool = True,
        pad_top: int = 0,
        pad_bottom: int = 0,
        pad_left: int = 0,
        pad_right: int = 0,
        save_prepared_input_path: str | Path | None = None,
        save_prepared_input_dtype: str | np.dtype = np.float32,
        save_prepared_input_normalized: bool = True,
    ):
        self.image_path = Path(image_path)
        if not self.image_path.exists():
            raise FileNotFoundError(self.image_path)

        self.backend = "memmap"
        self._zarr_store = None
        try:
            raw = tifffile.memmap(self.image_path)
        except ValueError:
            # Common for compressed/tiled TIFFs: try chunked zarr-backed reads.
            try:
                import zarr

                self._zarr_store = tifffile.imread(self.image_path, aszarr=True)
                raw = zarr.open(self._zarr_store, mode="r")
                self.backend = "zarr"
            except Exception:
                # Last resort fallback reads full array into memory.
                raw = tifffile.imread(self.image_path)
                self.backend = "ndarray"

        self.arr = raw
        if self.arr.ndim == 2:
            self.layout = "HW"
            self.channels = 1
            self.orig_h, self.orig_w = int(self.arr.shape[0]), int(self.arr.shape[1])
        elif self.arr.ndim == 3:
            if self.arr.shape[0] <= 8 and self.arr.shape[1] > 8 and self.arr.shape[2] > 8:
                self.layout = "CHW"
                self.channels = int(self.arr.shape[0])
                self.orig_h, self.orig_w = int(self.arr.shape[1]), int(self.arr.shape[2])
            else:
                self.layout = "HWC"
                self.channels = int(self.arr.shape[2])
                self.orig_h, self.orig_w = int(self.arr.shape[0]), int(self.arr.shape[1])
        else:
            raise ValueError(f"Expected 2D/3D TIFF, got shape {self.arr.shape}")
        self.spec = make_tile_grid_spec(
            orig_h=self.orig_h,
            orig_w=self.orig_w,
            patch_size=patch_size,
            stride=stride,
            pad_top=pad_top,
            pad_bottom=pad_bottom,
            pad_left=pad_left,
            pad_right=pad_right,
        )
        self.normalize = normalize

        mins: list[float] = []
        maxs: list[float] = []
        for ch in range(self.channels):
            if self.layout == "CHW":
                ch_data = np.asarray(self.arr[ch], dtype=np.float32)
            elif self.layout == "HWC":
                ch_data = np.asarray(self.arr[..., ch], dtype=np.float32)
            else:  # HW
                ch_data = np.asarray(self.arr, dtype=np.float32)
            mins.append(float(ch_data.min()))
            maxs.append(float(ch_data.max()))

        self.img_min = np.asarray(mins, dtype=np.float32)
        self.img_max = np.asarray(maxs, dtype=np.float32)

        if save_prepared_input_path is not None:
            self.save_prepared_input(
                out_path=save_prepared_input_path,
                dtype=save_prepared_input_dtype,
                normalized=save_prepared_input_normalized,
            )

    def _read_channel_full(self, ch: int) -> np.ndarray:
        if self.layout == "CHW":
            return np.asarray(self.arr[ch], dtype=np.float32)
        if self.layout == "HWC":
            return np.asarray(self.arr[..., ch], dtype=np.float32)
        return np.asarray(self.arr, dtype=np.float32)

    def _normalize_channel(self, ch_data: np.ndarray, ch: int) -> np.ndarray:
        return (ch_data - self.img_min[ch]) / (self.img_max[ch] - self.img_min[ch] + 1e-8)

    def get_prepared_input_array(
        self,
        normalized: bool | None = None,
        dtype: str | np.dtype = np.float32,
    ) -> np.ndarray:
        """Return full input array as (C,H,W) in requested dtype.

        This is useful for residual analysis against model predictions.
        """
        use_normalized = self.normalize if normalized is None else bool(normalized)
        out_dtype = _resolve_numpy_dtype(dtype)

        out = np.empty((self.channels, self.orig_h, self.orig_w), dtype=np.float32)
        for ch in range(self.channels):
            ch_data = self._read_channel_full(ch)
            if use_normalized:
                ch_data = self._normalize_channel(ch_data, ch)
            out[ch] = ch_data

        if np.issubdtype(out_dtype, np.integer):
            info = np.iinfo(out_dtype)
            out = np.clip(out, info.min, info.max)
        return out.astype(out_dtype, copy=False)

    def save_prepared_input(
        self,
        out_path: str | Path,
        dtype: str | np.dtype = np.float32,
        normalized: bool | None = None,
    ) -> Path:
        """Save full prepared input array (C,H,W) in requested dtype."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_dtype = _resolve_numpy_dtype(dtype)
        use_normalized = self.normalize if normalized is None else bool(normalized)

        mm = tifffile.memmap(str(out_path), shape=(self.channels, self.orig_h, self.orig_w), dtype=out_dtype)
        for ch in range(self.channels):
            ch_data = self._read_channel_full(ch)
            if use_normalized:
                ch_data = self._normalize_channel(ch_data, ch)
            if np.issubdtype(out_dtype, np.integer):
                info = np.iinfo(out_dtype)
                ch_data = np.clip(ch_data, info.min, info.max)
            mm[ch] = ch_data.astype(out_dtype, copy=False)
        mm.flush()
        return out_path

    def __len__(self) -> int:
        return self.spec.num_tiles_y * self.spec.num_tiles_x

    def _extract_patch_with_global_padding(self, top_padded: int, left_padded: int) -> np.ndarray:
        p = self.spec.patch_size

        src_top = top_padded - self.spec.pad_top
        src_left = left_padded - self.spec.pad_left
        src_bottom = src_top + p
        src_right = src_left + p

        in_top = max(src_top, 0)
        in_left = max(src_left, 0)
        in_bottom = min(src_bottom, self.orig_h)
        in_right = min(src_right, self.orig_w)

        if self.layout == "CHW":
            patch = np.asarray(self.arr[:, in_top:in_bottom, in_left:in_right], dtype=np.float32)
        elif self.layout == "HWC":
            patch = np.asarray(self.arr[in_top:in_bottom, in_left:in_right, :], dtype=np.float32)
            patch = np.transpose(patch, (2, 0, 1))
        else:  # HW
            patch = np.asarray(self.arr[in_top:in_bottom, in_left:in_right], dtype=np.float32)
            patch = patch[np.newaxis, ...]

        pad_t = in_top - src_top
        pad_l = in_left - src_left
        pad_b = src_bottom - in_bottom
        pad_r = src_right - in_right

        if pad_t or pad_b or pad_l or pad_r:
            mode = "reflect"
            if patch.shape[-2] < 2 or patch.shape[-1] < 2:
                mode = "edge"
            patch = np.pad(
                patch,
                ((0, 0), (pad_t, pad_b), (pad_l, pad_r)),
                mode=mode,
            )

        return patch.astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ty = idx // self.spec.num_tiles_x
        tx = idx % self.spec.num_tiles_x
        top = ty * self.spec.stride
        left = tx * self.spec.stride

        patch = self._extract_patch_with_global_padding(top, left)
        if self.normalize:
            pmin = self.img_min[:, np.newaxis, np.newaxis]
            pmax = self.img_max[:, np.newaxis, np.newaxis]
            patch = (patch - pmin) / (pmax - pmin + 1e-8)

        return {
            "input": torch.from_numpy(patch),
            "tile_y": torch.tensor(ty, dtype=torch.int64),
            "tile_x": torch.tensor(tx, dtype=torch.int64),
            "top": torch.tensor(top, dtype=torch.int64),
            "left": torch.tensor(left, dtype=torch.int64),
        }


def make_tiled_inference_loader(
    image_path: str | Path,
    patch_size: int,
    stride: int,
    batch_size: int,
    num_workers: int,
    normalize: bool = True,
    pad_top: int = 0,
    pad_bottom: int = 0,
    pad_left: int = 0,
    pad_right: int = 0,
    save_prepared_input_path: str | Path | None = None,
    save_prepared_input_dtype: str | np.dtype = np.float32,
    save_prepared_input_normalized: bool = True,
) -> tuple[DataLoader, TiffTiledInferenceDataset]:
    ds = TiffTiledInferenceDataset(
        image_path=image_path,
        patch_size=patch_size,
        stride=stride,
        normalize=normalize,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_left=pad_left,
        pad_right=pad_right,
        save_prepared_input_path=save_prepared_input_path,
        save_prepared_input_dtype=save_prepared_input_dtype,
        save_prepared_input_normalized=save_prepared_input_normalized,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, ds


class TiffAutoencoderDataset(Dataset):
    def __init__(self, files: Sequence[Path], image_size: int):
        self.files = list(files)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self.files[idx]
        img = tifffile.imread(path)
        x = _to_chw_float32(img)
        x = _center_crop(x, self.image_size)
        x = _per_image_minmax(x)
        return {"input": x, "target": x, "path": str(path)}


def split_train_val(files: Sequence[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    n_val = max(1, int(len(files) * val_fraction))
    val_idx = set(idx[:n_val].tolist())
    train_files = [f for i, f in enumerate(files) if i not in val_idx]
    val_files = [f for i, f in enumerate(files) if i in val_idx]
    return train_files, val_files


def make_dataloaders(
    train_dir: str,
    val_dir: str | None,
    image_size: int,
    batch_size: int,
    num_workers: int,
    val_fraction_if_no_val_dir: float,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_files = _list_tiff_files(Path(train_dir))

    if val_dir:
        val_files = _list_tiff_files(Path(val_dir))
    else:
        train_files, val_files = split_train_val(train_files, val_fraction_if_no_val_dir, seed)

    train_ds = TiffAutoencoderDataset(train_files, image_size=image_size)
    val_ds = TiffAutoencoderDataset(val_files, image_size=image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader