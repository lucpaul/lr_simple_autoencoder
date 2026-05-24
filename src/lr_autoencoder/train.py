from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from .config import FullConfig, load_config
from .data import make_dataloaders
from .model import SimpleUNetAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple UNet autoencoder on LR TIFF images")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    amp: bool,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_items = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, leave=False):
            x = batch["input"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                pred = model(x)
                loss = criterion(pred, y)

            if is_train:
                assert optimizer is not None
                if scaler is not None and amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_items += bs

    return total_loss / max(total_items, 1)


def _save_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: FullConfig,
    epoch: int,
    train_loss: float,
    val_loss: float,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_config": {
                "in_channels": cfg.train.in_channels,
                "base_channels": cfg.train.base_channels,
            },
            "data_config": {
                "image_size": cfg.data.image_size,
            },
        },
        ckpt_path,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    _set_seed(cfg.data.seed)

    output_dir = Path(cfg.train.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / "metrics.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = make_dataloaders(
        train_dir=cfg.data.train_dir,
        val_dir=cfg.data.val_dir,
        image_size=cfg.data.image_size,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        val_fraction_if_no_val_dir=cfg.data.val_fraction_if_no_val_dir,
        seed=cfg.data.seed,
    )

    model = SimpleUNetAutoencoder(
        in_channels=cfg.train.in_channels,
        base_channels=cfg.train.base_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp and device.type == "cuda")

    best_val = float("inf")
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

        for epoch in range(1, cfg.train.epochs + 1):
            train_loss = _run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                amp=cfg.train.amp,
            )
            val_loss = _run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
                scaler=None,
                amp=False,
            )

            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}"])
            f.flush()

            print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

            _save_checkpoint(ckpt_dir / "last.pt", model, optimizer, cfg, epoch, train_loss, val_loss)

            if val_loss < best_val:
                best_val = val_loss
                _save_checkpoint(ckpt_dir / "best.pt", model, optimizer, cfg, epoch, train_loss, val_loss)


if __name__ == "__main__":
    main()