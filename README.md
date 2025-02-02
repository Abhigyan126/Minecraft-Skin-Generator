<h2 align='center'>
Minecraft Skin Generator using WGAN-GP
</h2>
A deep learning model that generates unique Minecraft character skins using a Wasserstein GAN with Gradient Penalty (WGAN-GP).

## implementation

<p align="center">Check out Nextjs implementation: https://blockface.vercel.app/</p>

## Sample

<p float="left">
  
![render png 09-47-12-210](https://github.com/user-attachments/assets/6c86e1ea-d5c0-458f-9d4a-05b5f0967e69)
![render](https://github.com/user-attachments/assets/0a4fa3c1-4a69-4a51-b780-9d155a80eb76)
![render-2](https://github.com/user-attachments/assets/e2e94b8a-0fb4-464a-8a73-7ca42f6121f8)
![render-3](https://github.com/user-attachments/assets/0323d15b-5936-4aeb-9ace-72422d9e042a)
![render-4](https://github.com/user-attachments/assets/fa5f47c1-3def-4245-8e19-5658b8bcb33d)
![render-5](https://github.com/user-attachments/assets/a807aee0-eed0-4282-83a7-81e579a25203)
![render-6](https://github.com/user-attachments/assets/3ac69eec-f484-4d53-ae51-b904b5a7df3c)
![render-8](https://github.com/user-attachments/assets/a8189543-f182-4525-ad54-6f7e2f97e4c1)
![render-9](https://github.com/user-attachments/assets/6ff1ed7f-7b28-4f79-a120-80eedee0a403)

</p>

## Overview

This project implements a generative adversarial network to create unique Minecraft character skins. It uses advanced techniques including self-attention mechanisms, adaptive residual blocks, and spectral normalization to produce high-quality, diverse character skins.

## Arcitecture
```mermaid
graph TD
    subgraph Data_Pipeline
        A[Raw Minecraft Skins] -->|Preprocessing| B[MinecraftImageDataset]
        B -->|Validation| C{Image Check}
        C -->|Valid| D[DataLoader]
        C -->|Invalid| E[Error Log]
        D -->|Batch Processing| F[Normalized Tensors]
    end

    subgraph Generator_Architecture
        G[Latent Space] -->|Dense Layer| H[Initial Features]
        H -->|Reshape| I[4x4 Feature Maps]
        I -->|Block 1| J[8x8 Features + Self-Attention]
        J -->|Block 2| K[16x16 Features + Self-Attention]
        K -->|Block 3| L[32x32 Features]
        L -->|Block 4| M[64x64 Features]
        M -->|Final Conv| N[Generated Skin]
        
        subgraph Residual_Block
            R1[Input] -->|Conv1| R2[Norm + LeakyReLU]
            R2 -->|Conv2| R3[Norm]
            R1 -->|Shortcut| R4[1x1 Conv]
            R3 -->|SE Block| R5[Channel Recalibration]
            R4 --> R6[Add]
            R5 --> R6
        end
    end

    subgraph Critic_Architecture
        O[Input Image] -->|Initial Block| P[Feature Maps]
        P -->|Block 1| Q[Deep Features + Self-Attention]
        Q -->|Block 2| S[Deeper Features]
        S -->|Block 3| T[Final Features]
        T -->|Flatten| U[Vector]
        U -->|Linear| V[Criticism Score]
    end

    subgraph Training_Loop
        W[Real & Fake Pairs] -->|Forward Pass| X[Loss Computation]
        X -->|Gradient Penalty| Y[WGAN-GP Loss]
        Y -->|Backprop| Z[Parameter Updates]
        Z -->|New Batch| W
    end

    subgraph Evaluation_Metrics
        AA[Generated Samples] -->|SSIM| AB[Structural Similarity]
        AA -->|Visual Check| AC[Sample Gallery]
        AB -->|Tracking| AD[Loss Curves]
        AC -->|Save| AE[Sample Archive]
    end

    subgraph Post_Processing
        AF[Raw Generated Skin] -->|Format Convert| AG[PNG Output]
        AG -->|SkinPy| AH[3D Render]
        AH -->|Multiple Views| AI[Final Visualization]
    end

    F -->|Real Images| W
    N -->|Fake Images| W
    N -->|Output| AA
    V -->|Feedback| X
```

## Features

- WGAN-GP architecture with gradient penalty for stable training
- Self-attention mechanisms for better global coherence
- Adaptive residual blocks with squeeze-and-excitation
- Spectral normalization for improved training stability
- SSIM (Structural Similarity Index) monitoring during training
- Multi-GPU support for faster training
- Custom dataset handling with image validation


### Environment Setup

It's recommended to use Python 3.11.x with Conda:

```bash
conda create -n minecraft-gan python=3.11
conda activate minecraft-gan

pip install -r requirements.txt
```

## Project Structure

```
minecraft-skin-generator/
├── test/
│   ├── save/
│   │   └── *.png
│   └── *.py
├── model/
│   ├── docs/
│   │   └── *.txt
│   ├── samples/
│   │   ├── model v5/
│   │   │   └── *.png
│   │   └── model v3/
│   │       └── *.png
│   └── *.pth
├── train/
│   └── *.py
├── ipynb_files/
│   └── *.ipynb
├── generated_skins/
│   ├── *.txt
│   └── *.png
└── README.md
```

## Usage

```bash 
# for training
python train/train.py

# for testing
python test/test.py
```

## Files Description

### Training Files (`train/`)
- Main training scripts for the GAN
- Data loading and preprocessing utilities
- Model configuration and hyperparameter settings

### Model Files (`model/`)
- Saved model checkpoints (`.pth`)
- Sample outputs in version-specific directories
- Documentation and model architecture details

### Testing Files (`test/`)
- Scripts for generating and testing skins
- Save directory for test outputs
- Validation utilities

### Jupyter Notebooks (`ipynb_files/`)
- Development and experimentation notebooks
- Training visualization and analysis

### Generated Skins (`generated_skins/`)
- Output directory for generated sample Minecraft skins
- Associated metadata

## Monitoring and Evaluation

The training process includes:
- Generator and Critic loss tracking
- SSIM score monitoring
- Regular sample generation for visual inspection

