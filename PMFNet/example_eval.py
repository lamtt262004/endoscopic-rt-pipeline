"""
Example testing/evaluation script for PMF-Net
Shows how to evaluate the model on test datasets
"""
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np

from models import pvt_v2_b2, PCRN
from training import Segmentor
from utils import PolypDS, get_augmentation_transforms, iou_score_fn, dice_coef_fn


def evaluate_model(model, test_loader, dataset_name):
    """
    Evaluate model on a test dataset
    
    Args:
        model: PCRN model
        test_loader: Test data loader
        dataset_name: Name of dataset for logging
    """
    model.eval()
    
    dice_scores = []
    iou_scores = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.cuda()
            masks = masks.cuda()
            
            outputs = model(images)
            
            dice = dice_coef_fn(outputs, masks)
            iou = iou_score_fn(outputs, masks)
            
            dice_scores.append(dice)
            iou_scores.append(iou)
    
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    
    print(f"\n{dataset_name}:")
    print(f"  Dice Score: {avg_dice:.4f}")
    print(f"  IoU Score: {avg_iou:.4f}")
    
    return avg_dice, avg_iou


def main():
    # Configuration
    POLYP_DATA_PATH = 'path/to/polypData.npz'
    CHECKPOINT_PATH = 'checkpoints/ckpt_XXXX.ckpt'  # Update with your checkpoint
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data augmentation (use validation transform for testing)
    _, val_transform = get_augmentation_transforms(image_size=256)
    
    # Test datasets
    test_datasets = {
        'Kvasir-SEG': PolypDS(data_path=POLYP_DATA_PATH, type='test_kvasir', transform=val_transform),
        'ETIS-LaribPolypDB': PolypDS(data_path=POLYP_DATA_PATH, type='test_etis', transform=val_transform),
        'CVC-300': PolypDS(data_path=POLYP_DATA_PATH, type='test_cvc300', transform=val_transform),
        'CVC-ClinicDB': PolypDS(data_path=POLYP_DATA_PATH, type='test_clinic', transform=val_transform),
        'Colon-DB': PolypDS(data_path=POLYP_DATA_PATH, type='test_colon', transform=val_transform),
    }
    
    # Load model
    print("Loading model...")
    segmentor = Segmentor.load_from_checkpoint(CHECKPOINT_PATH)
    model = segmentor.model.to(device)
    
    # Evaluate on all test datasets
    print("Evaluating model...")
    results = {}
    
    for dataset_name, dataset in test_datasets.items():
        test_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)
        dice, iou = evaluate_model(model, test_loader, dataset_name)
        results[dataset_name] = {'dice': dice, 'iou': iou}
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    avg_dice = np.mean([r['dice'] for r in results.values()])
    avg_iou = np.mean([r['iou'] for r in results.values()])
    
    for dataset_name, scores in results.items():
        print(f"{dataset_name:20s} - Dice: {scores['dice']:.4f}, IoU: {scores['iou']:.4f}")
    
    print(f"{'Average':20s} - Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")


if __name__ == "__main__":
    main()
