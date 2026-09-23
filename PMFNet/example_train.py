"""
Example training script for PMF-Net
Shows how to train the model on your dataset
"""
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from models import pvt_v2_b2, PCRN
from training import Segmentor, HistoryLogger
from utils import PolypDS, get_augmentation_transforms


def main():
    # Configuration
    POLYP_DATA_PATH = 'path/to/polypData.npz'
    PRETRAINED_BACKBONE_PATH = 'path/to/pvt_v2_b2.pth'
    CHECKPOINT_DIR = 'checkpoints/'
    BATCH_SIZE = 16
    MAX_EPOCHS = 150
    LEARNING_RATE = 1e-4
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data augmentation
    train_transform, val_transform = get_augmentation_transforms(image_size=256)
    
    # Dataset
    print("Loading dataset...")
    train_ds = PolypDS(data_path=POLYP_DATA_PATH, type='train', transform=train_transform)
    val_ds = PolypDS(data_path=POLYP_DATA_PATH, type='val', transform=val_transform)
    
    # DataLoader
    trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=8, num_workers=2, shuffle=False)
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    
    # Model
    print("Building model...")
    backbone = pvt_v2_b2()
    backbone.load_state_dict(torch.load(PRETRAINED_BACKBONE_PATH, map_location=device))
    
    model = PCRN(backbone=backbone)
    model.to(device)
    
    # Lightning module
    segmentor = Segmentor(model=model)
    
    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="ckpt_{val_dice:0.4f}",
        monitor="val_dice",
        mode="max",
        save_top_k=1,
        verbose=True,
        save_weights_only=True,
        auto_insert_metric_name=False,
    )
    
    progress_bar = pl.callbacks.TQDMProgressBar()
    history_logger = HistoryLogger(dir=f"{CHECKPOINT_DIR}/history.csv")
    
    # Trainer configuration
    trainer_config = {
        "max_epochs": MAX_EPOCHS,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1 if torch.cuda.is_available() else None,
        "callbacks": [checkpoint_callback, progress_bar, history_logger],
        "logger": False,
        "log_every_n_steps": 1,
        "num_sanity_val_steps": 0,
        "precision": 16 if torch.cuda.is_available() else 32,
        "benchmark": True,
        "enable_progress_bar": True,
    }
    
    # Trainer
    trainer = pl.Trainer(**trainer_config)
    
    # Train
    print("Starting training...")
    trainer.fit(segmentor, trainloader, valloader)
    
    print("Training completed!")
    print(f"Best model saved to {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
