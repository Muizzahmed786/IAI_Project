import torch
import torch.nn as nn
import torch.optim as optim
from models.conditionalGenerator import Generator
from models.conditionalDiscriminator import Discriminator
from utils.dataset import get_dataloader
import torchvision.utils as vutils
import os

def gradient_penalty(critic, labels, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, labels)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z_dim = 100
    lr = 1e-4 # WGAN usually performs better with lower LR
    batch_size = 64
    lambda_gp = 10 # Coefficient for Gradient Penalty

    gen = Generator(z_dim).to(device)
    critic = Discriminator().to(device)
    
    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))

    loader, _ = get_dataloader("data/train", batch_size=batch_size)

    # Assume your loader returns (images, labels)
    # loader, _ = get_dataloader("data/train", batch_size)

    for epoch in range(50):
        for real, labels in loader:
            real = real.to(device)
            labels = labels.to(device)
            cur_batch_size = real.shape[0]

            # --- Train Critic (5 times for every 1 Gen update is standard for WGAN) ---
            for _ in range(5):
                noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
                fake = gen(noise, labels)
                critic_real = critic(real, labels).reshape(-1)
                critic_fake = critic(fake, labels).reshape(-1)
                gp = gradient_penalty(critic, labels, real, fake, device)
                # WGAN Loss: E[critic(fake)] - E[critic(real)] + GP
                loss_critic = (torch.mean(critic_fake) - torch.mean(critic_real)) + (lambda_gp * gp)
                
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            # --- Train Generator ---
            gen_fake = critic(fake, labels).reshape(-1)
            loss_gen = -torch.mean(gen_fake)

            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

        print(f"Epoch {epoch} | Critic Loss: {loss_critic:.4f} | Gen Loss: {loss_gen:.4f}")

if __name__ == "__main__":
    train()