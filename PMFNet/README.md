# PMF-Net

A PyTorch implementation of **PMF-Net** - Polyp segmentation network with Multi-scale Feature fusion and Pyramid Vision Transformer backbone.

## Overview

This repository contains the implementation of a polyp segmentation network that combines:
- **PVT V2 (Pyramid Vision Transformer V2)** as the backbone encoder
- **Channel Layer (Ch-layer)** for channel attention
- **Group Aggregation Bridge (GAB)** modules for feature fusion
- **Multi-scale Decoder** with boundary-guided attention

## Architecture

The network consists of:

### 1. **Backbone** (models/backbone.py)
- Pyramid Vision Transformer V2 (PVT V2) variants: B0, B1, B2, B3, B4, B5
- Multi-scale feature extraction with patch embeddings
- Spatial reduction attention for efficient computation

### 2. **Blocks** (models/blocks.py)
- **ChLayer**: Channel Attention Layer
- **CBAM**: Convolutional Block Attention Module
- **BFEB**: Boundary Feature Enhancement Block
- **GAB**: Group Aggregation Bridge Module
- **PASPP**: Pyramid Atrous Spatial Pyramid Pooling

### 3. **Decoder** (models/decoder.py)
- Progressive upsampling with skip connections
- Boundary attention mask generation
- Foreground-background separation

### 4. **Main Model** (models/pcrn.py)
- **PCRN**: Complete Polyp Convolutional Residual Network
- Integrates all components with residual connections

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PMF-Net.git
cd PMF-Net

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Loading a Pre-trained Model

```python
import torch
from models import pvt_v2_b2, PCRN

# Load backbone
backbone = pvt_v2_b2()
backbone.load_state_dict(torch.load('path/to/pvt_v2_b2.pth'))

# Create full model
model = PCRN(backbone=backbone)
model.load_state_dict(torch.load('path/to/checkpoint.pth'))
```

### Inference

```python
import torch
from torchvision import transforms
from PIL import Image

# Load model
model = PCRN(backbone=backbone)
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                        std=(0.229, 0.224, 0.225))
])

image = Image.open('image.jpg')
image = transform(image).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(image)
    prediction = (output > 0.5).squeeze().numpy()
```

### Training

```python
import pytorch_lightning as pl
from training import Segmentor, HistoryLogger
from torch.utils.data import DataLoader

# Create model and lightning module
model = PCRN(backbone=backbone)
segmentor = Segmentor(model=model)

# Setup data loaders
trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Configure trainer
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
    callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_dice", mode="max")],
    logger=False
)

# Train
trainer.fit(segmentor, trainloader, valloader)
```

## Dataset

The model uses the polyp segmentation dataset with different test sets:
- **Kvasir-SEG**: Endoscopic polyp segmentation
- **ETIS-LaribPolypDB**: Magnified polyp segmentation
- **CVC-300**: Colonoscopy polyp segmentation
- **CVC-ClinicDB**: Clinical polyp segmentation
- **Colon-DB**: Histopathological polyp segmentation

### Data Format

The dataset should be in NPZ format with the following structure:
```python
{
    'train_img': (N, H, W, 3),  # Training images
    'train_msk': (N, H, W, 1),  # Training masks
    'val_img':   (N, H, W, 3),  # Validation images
    'val_msk':   (N, H, W, 1),  # Validation masks
    'test_*_img': (N, H, W, 3), # Test images
    'test_*_msk': (N, H, W, 1)  # Test masks
}
```

## Evaluation Metrics

- **Dice Coefficient**: Measures overlap between prediction and ground truth
- **IoU Score**: Intersection over Union metric
- **Tversky Index**: Weighted metric emphasizing false positives/negatives

## Project Structure

```
PMF-Net/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── models/
│   ├── __init__.py
│   ├── backbone.py                   # PVT V2 backbone
│   ├── blocks.py                     # Attention blocks
│   ├── decoder.py                    # Decoder module
│   └── pcrn.py                       # Main PCRN model
├── utils/
│   ├── __init__.py
│   ├── metrics.py                    # Evaluation metrics
│   ├── losses.py                     # Loss functions
│   └── dataset.py                    # Dataset classes
├── training/
│   ├── __init__.py
│   ├── segmentor.py                  # PyTorch Lightning module
│   └── callbacks.py                  # Training callbacks
└── notebooks/
    └── PVTV2B2_NewGABmodule_Newdecoder(Final).ipynb  # Original notebook
```

## Performance

The model achieves competitive results on polyp segmentation benchmarks:

| Dataset | Dice | IoU |
|---------|------|-----|
| Kvasir-SEG | 0.93+ | 0.87+ |
| ETIS-LaribPolypDB | 0.91+ | 0.84+ |
| CVC-300 | 0.92+ | 0.85+ |




## Acknowledgments

- PVT V2: [Pyramid Vision Transformer V2](https://github.com/whai362/PVT)
- Inspired by recent advances in medical image segmentation
- CBAM: [Convolutional Block Attention Module](https://github.com/Jongchan/attention-is-all-you-need-pytorch)

## Contact

For questions or issues, please open an issue on GitHub or contact the author.
