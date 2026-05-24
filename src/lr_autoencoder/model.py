from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, conv_padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=conv_padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=conv_padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleUNetAutoencoder(nn.Module):
    """3 downsampling stages using stride-2 convolutions, no pooling."""

    def __init__(self, in_channels: int = 4, base_channels: int = 32, conv_padding: int = 1):
        super().__init__()

        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )

        self.enc1 = ConvBlock(in_channels, c1, conv_padding=conv_padding)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=conv_padding)
        self.enc2 = ConvBlock(c2, c2, conv_padding=conv_padding)

        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=conv_padding)
        self.enc3 = ConvBlock(c3, c3, conv_padding=conv_padding)

        self.down3 = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=conv_padding)
        self.bottleneck = ConvBlock(c4, c4, conv_padding=conv_padding)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 + c3, c3, conv_padding=conv_padding)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2, conv_padding=conv_padding)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1, conv_padding=conv_padding)

        self.out = nn.Conv2d(c1, in_channels, kernel_size=1)

    @staticmethod
    def _center_crop_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Center-crop src to ref spatial size for robust skip concatenation."""
        _, _, h, w = src.shape
        _, _, rh, rw = ref.shape
        if h == rh and w == rw:
            return src
        top = max((h - rh) // 2, 0)
        left = max((w - rw) // 2, 0)
        return src[:, :, top : top + rh, left : left + rw]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        xb = self.bottleneck(self.down3(x3))

        y3 = self.up3(xb)
        x3 = self._center_crop_like(x3, y3)
        y3 = self.dec3(torch.cat([y3, x3], dim=1))

        y2 = self.up2(y3)
        x2 = self._center_crop_like(x2, y2)
        y2 = self.dec2(torch.cat([y2, x2], dim=1))

        y1 = self.up1(y2)
        x1 = self._center_crop_like(x1, y1)
        y1 = self.dec1(torch.cat([y1, x1], dim=1))

        return self.out(y1)