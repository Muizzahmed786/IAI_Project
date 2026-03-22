import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, num_classes=3, img_channels=1):
        super().__init__()
        # Embed label into a vector of size z_dim
        self.label_embed = nn.Embedding(num_classes, z_dim)
        
        self.gen = nn.Sequential(
            # Input: noise + label embedding (z_dim * 2)
            nn.ConvTranspose2d(z_dim * 2, 1024, 4, 1, 0), 
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            nn.ConvTranspose2d(1024, 512, 4, 2, 1), 
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, img_channels, 4, 2, 1),
            nn.Tanh() # Output: (batch, 1, 128, 128)
        )

    def forward(self, noise, labels):
        # Reshape label embedding to (batch, z_dim, 1, 1)
        label_vec = self.label_embed(labels).unsqueeze(2).unsqueeze(3)
        # Concatenate noise and label embedding
        x = torch.cat([noise, label_vec], dim=1)
        return self.gen(x)