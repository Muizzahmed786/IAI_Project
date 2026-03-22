import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channels=1):
        super().__init__()

        self.disc = nn.Sequential(

            # Input: (batch, 1, 128, 128)

            nn.Conv2d(img_channels, 64, 4, 2, 1),   # 64x64
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),           # 32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),          # 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 4, 2, 1),          # 8x8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1024, 4, 2, 1),         # 4x4
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),

            nn.Conv2d(1024, 1, 4, 1, 0),           # 1x1
        )

    def forward(self, x):
        return self.disc(x)