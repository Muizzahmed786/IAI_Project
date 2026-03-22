import torch
import torch.nn as nn
import torch.optim as optim
from models.generator import Generator
from models.discriminator import Discriminator
from utils.dataset import get_dataloader
import torchvision.utils as vutils
import os


def main():

    device = torch.device("cuda")
    print("Using device:", device)

    # Hyperparameters
    lr = 0.0002
    batch_size = 64
    z_dim = 100
    epochs = 51

    # Create folder
    os.makedirs("generated", exist_ok=True)

    # Load dataset
    loader, classes = get_dataloader("data/train", batch_size=batch_size)

    # Models
    gen = Generator(z_dim).to(device)
    disc = Discriminator().to(device)

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    # Optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=0.0001, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(epochs):

        for real, _ in loader:

            real = real.to(device, non_blocking=True)
            cur_batch_size = real.shape[0]

            # Train Discriminator
            noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
            fake = gen(noise)

            disc_real = disc(real).reshape(-1)
            loss_real = criterion(disc_real, torch.ones_like(disc_real) * 0.9)

            disc_fake = disc(fake.detach()).reshape(-1)
            loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

            loss_disc = (loss_real + loss_fake) / 2

            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Train Generator
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))

            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

        # Save images every 5 epochs
        if epoch % 5 == 0:
            with torch.no_grad():
                fake = gen(torch.randn(16, z_dim, 1, 1).to(device))
                vutils.save_image(fake, f"generated/epoch_{epoch}.png", normalize=True)

        print(f"Epoch [{epoch}/{epochs}] | D Loss: {loss_disc:.4f} | G Loss: {loss_gen:.4f}")


# VERY IMPORTANT LINE (fixes your error)
if __name__ == "__main__":
    main()