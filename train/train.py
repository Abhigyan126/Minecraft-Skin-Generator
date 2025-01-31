from generator import Generatorv5, Generatorv3
from critic import Criticv2, Criticv1
from data_loder import load_minecraft_data
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim_sklearn

# Gradient Penalty Function
def compute_gradient_penalty(critic, real_imgs, fake_imgs, device):
    """Compute the gradient penalty for WGAN-GP."""
    alpha = torch.rand(real_imgs.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = torch.ones(real_imgs.size(0), 1).to(device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# WGAN Training
def train_wgan(generator, critic, dataloader, latent_dim, epochs, sample_interval=1, lambda_gp=10):
    optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.999))
    optimizer_C = optim.Adam(critic.parameters(), lr=4e-4, betas=(0.0, 0.999))

    g_losses, c_losses, ssim_scores = [], [], []

    for epoch in range(epochs):
        g_loss_epoch, c_loss_epoch = 0.0, 0.0
        epoch_ssim_scores = []

        for i, imgs in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
            real_imgs = imgs.to(device).float()
            batch_size = real_imgs.size(0)

            # Train Critic
            optimizer_C.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z)
            real_loss = torch.mean(critic(real_imgs))
            fake_loss = torch.mean(critic(fake_imgs))
            gradient_penalty = compute_gradient_penalty(critic, real_imgs, fake_imgs, device)
            c_loss = fake_loss - real_loss + lambda_gp * gradient_penalty
            c_loss.backward()
            optimizer_C.step()
            c_loss_epoch += c_loss.item()

            # Train Generator every n_critic steps
            if i % 5 == 0:
                optimizer_G.zero_grad()
                z = torch.randn(batch_size, latent_dim).to(device)
                gen_imgs = generator(z)
                g_loss = -torch.mean(critic(gen_imgs))
                g_loss.backward()
                optimizer_G.step()
                g_loss_epoch += g_loss.item()

                # Calculate SSIM periodically
                ssim_score = calculate_ssim(real_imgs, gen_imgs)
                epoch_ssim_scores.append(ssim_score)

        # Record losses and SSIM
        g_losses.append(g_loss_epoch / len(dataloader))
        c_losses.append(c_loss_epoch / len(dataloader))
        ssim_scores.append(np.mean(epoch_ssim_scores))

        # Save sample images
        if (epoch + 1) % sample_interval == 0:
            display_sample_images(generator, latent_dim)

        print(f"[Epoch {epoch + 1}] Generator Loss: {g_losses[-1]:.4f}, Critic Loss: {c_losses[-1]:.4f}, SSIM: {ssim_scores[-1]:.4f}")

    # Plot losses and SSIM
    plot_losses_and_ssim(g_losses, c_losses, ssim_scores)

# Display and Plot Functions
def display_sample_images(generator, latent_dim):
    generator.eval()
    z = torch.randn(16, latent_dim).to(device)
    gen_imgs = generator(z).detach().cpu()
    gen_imgs = (gen_imgs + 1) / 2  # Rescale to [0, 1]
    grid = make_grid(gen_imgs, nrow=4)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()
    generator.train()

def plot_losses_and_ssim(g_losses, c_losses, ssim_scores):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Loss plot
    ax1.plot(g_losses, label="Generator Loss")
    ax1.plot(c_losses, label="Critic Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    
    # SSIM plot
    ax2.plot(ssim_scores, label="SSIM Score", color='green')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("SSIM")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
def calculate_ssim(real_imgs, gen_imgs, win_size=3):
    """
    Calculate SSIM between real and generated images
    
    Args:
    real_imgs (torch.Tensor): Real image batch
    gen_imgs (torch.Tensor): Generated image batch
    win_size (int): Window size for SSIM calculation (should be smaller than the image size)
    
    Returns:
    float: Average SSIM score
    """
    # Convert images to numpy for SSIM calculation
    real_np = real_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1)
    gen_np = gen_imgs.detach().cpu().numpy().transpose(0, 2, 3, 1)
    
    # Normalize images to [0, 1] range
    real_np = (real_np + 1) / 2
    gen_np = (gen_np + 1) / 2
    
    # Compute SSIM for each image with custom window size and data_range specified
    ssim_scores = [ssim_sklearn(real_np[i], gen_np[i], multichannel=True, win_size=win_size, data_range=1.0) 
                   for i in range(len(real_np))]
    
    return np.mean(ssim_scores)

# Hyperparameters and Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 128
img_channels = 3
img_size = 64
epochs = 500
batch_size = 64
lambda_gp = 10

# uncomment to train model v5
generator = Generatorv5(latent_dim, img_channels).to(device)
critic = Criticv2(img_channels).to(device)

# uncomment to train model v3
# generator = Generatorv3(latent_dim, img_channels).to(device)
# critic = Criticv1(img_channels).to(device)


data_path = "path-to-data-dir"  # Replace with your dataset directory
dataloader = load_minecraft_data(data_path, img_size=img_size, batch_size=batch_size)

 # Initialize and Train WGAN
train_wgan(generator, critic, dataloader, latent_dim, epochs, sample_interval=5, lambda_gp=lambda_gp)
torch.save(generator.state_dict(), f"generator_epoch_{epochs}.pth")
torch.save(critic.state_dict(), f"critic_epoch_{epochs}.pth")