import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from generator import SelfAttention

# Critic Model (Discriminator)
class Criticv2(nn.Module):
    def __init__(self, img_channels, features_d=64):
        super(Criticv2, self).__init__()
        
        def critic_block(in_channels, out_channels, normalize=True):
            layers = [nn.utils.spectral_norm(
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
            )]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.features_d = features_d
        
        self.initial = nn.Sequential(
            *critic_block(img_channels, features_d, normalize=False),
            SelfAttention(features_d)
        )
        
        self.main = nn.Sequential(
            *critic_block(features_d, features_d * 2),
            SelfAttention(features_d * 2),
            *critic_block(features_d * 2, features_d * 4),
            SelfAttention(features_d * 4),
            *critic_block(features_d * 4, features_d * 8),
        )
        
        # Calculate the correct input size for the final linear layer
        # After 4 downsampling layers (stride=2), the spatial dimensions are reduced by factor of 16
        # For 64x64 input: 64/16 = 4, so final spatial dimensions are 4x4
        final_spatial_size = 4
        flattened_size = features_d * 8 * final_spatial_size * final_spatial_size
        
        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 1)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, img):
        features = self.initial(img)
        features = self.main(features)
        return self.final(features)
    
# Critic Model (Discriminator)
class Criticv1(nn.Module):
    def __init__(self, img_channels):
        super(Criticv1, self).__init__()
        def critic_block(in_channels, out_channels):
            return [
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        self.model = nn.Sequential(
            *critic_block(img_channels, 64),
            *critic_block(64, 128),
            *critic_block(128, 256),
            
            *critic_block(256, 512),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),  # Output Wasserstein distance
        )

    def forward(self, img):
        return self.model(img)