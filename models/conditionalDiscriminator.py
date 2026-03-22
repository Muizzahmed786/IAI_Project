import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, num_classes=3, img_channels=1, img_size=128):
        super().__init__()
        self.img_size = img_size
        # Embed label into an image-sized mask (1, 128, 128)
        self.label_embed = nn.Embedding(num_classes, img_size * img_size)

        self.disc = nn.Sequential(
            # Input: image + label mask (img_channels + 1)
            nn.Conv2d(img_channels + 1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128, affine=True), # WGAN-GP prefers InstanceNorm
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(1024, 1, 4, 1, 0), # Output is a score, no Sigmoid!
        )

    def forward(self, x, labels):
        # Create label mask: (batch, 1, 128, 128)
        label_mask = self.label_embed(labels).view(-1, 1, self.img_size, self.img_size)
        # Concatenate image and label mask
        x = torch.cat([x, label_mask], dim=1)
        return self.disc(x)