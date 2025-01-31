import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


# Generator Model
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, height * width)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        proj_value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        return self.gamma * out + x

class AdaptiveResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_spectral=True):
        super(AdaptiveResidualBlock, self).__init__()
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
        # Conditional batch norm or instance norm based on input size
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        
        # Optional spectral normalization
        if use_spectral:
            self.conv1 = nn.utils.spectral_norm(
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False)
            )
            self.conv2 = nn.utils.spectral_norm(
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        
        # Squeeze-and-Excitation block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, 1),
            nn.Sigmoid()
        )
        
        # Residual connection
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        # Apply squeeze-and-excitation
        se_weight = self.se(out)
        out = out * se_weight
        
        if self.shortcut is not None:
            residual = self.shortcut(residual)
            
        out += residual
        out = self.activation(out)
        
        return out

class Generatorv5(nn.Module):
    def __init__(self, latent_dim, img_channels, features_g=64):
        super(Generatorv5, self).__init__()
        self.init_size = 4
        self.latent_dim = latent_dim
        
        # Calculate proper size for initial dense layer
        # features_g * 16 represents the number of feature maps in the first layer
        self.initial = nn.Sequential(
            # Changed dimension to match the expected size
            nn.Linear(latent_dim, features_g * 16 * self.init_size * self.init_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Main generation blocks
        self.main = nn.ModuleList([
            # 4x4 -> 8x8
            nn.Sequential(
                AdaptiveResidualBlock(features_g * 16, features_g * 8),
                nn.Upsample(scale_factor=2),
                SelfAttention(features_g * 8)
            ),
            # 8x8 -> 16x16
            nn.Sequential(
                AdaptiveResidualBlock(features_g * 8, features_g * 4),
                nn.Upsample(scale_factor=2),
                SelfAttention(features_g * 4)
            ),
            # 16x16 -> 32x32
            nn.Sequential(
                AdaptiveResidualBlock(features_g * 4, features_g * 2),
                nn.Upsample(scale_factor=2)
            ),
            # 32x32 -> 64x64
            nn.Sequential(
                AdaptiveResidualBlock(features_g * 2, features_g),
                nn.Upsample(scale_factor=2)
            )
        ])
        
        # Final layers
        self.final = nn.Sequential(
            nn.Conv2d(features_g, img_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, z):
        # Initial dense layer
        out = self.initial(z)
        # Reshape using the correct dimensions
        out = out.view(-1, 64 * 16, self.init_size, self.init_size)
        
        # Main generation blocks
        for block in self.main:
            out = block(out)
        
        # Final convolution
        img = self.final(out)
        return img
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If in_channels and out_channels differ, use a 1x1 conv for shortcut
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        out += residual
        out = self.relu(out)

        return out


class Generatorv3(nn.Module):
    def __init__(self, latent_dim, img_channels):
        super(Generatorv3, self).__init__()
        self.init_size = 4  # Initial image size (4x4)
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 512 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2),  # 8x8
            ResidualBlock(512, 256),
            nn.Upsample(scale_factor=2),  # 16x16
            ResidualBlock(256, 128),
            nn.Upsample(scale_factor=2),  # 32x32
            ResidualBlock(128, 64),
            nn.Upsample(scale_factor=2),  # 64x64
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )


    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
