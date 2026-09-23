"""
Example inference script for PMF-Net polyp segmentation
"""
import torch
import numpy as np
import cv2
from pathlib import Path
from models import pvt_v2_b2, PCRN


def load_model(backbone_path, model_path, device='cuda'):
    """
    Load the pre-trained model
    
    Args:
        backbone_path: Path to pre-trained PVT V2 B2 weights
        model_path: Path to pre-trained PCRN weights
        device: Device to load model on
    
    Returns:
        model: Loaded PCRN model
    """
    # Load backbone
    backbone = pvt_v2_b2()
    backbone.load_state_dict(torch.load(backbone_path, map_location=device))
    
    # Create model
    model = PCRN(backbone=backbone)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_path, size=256):
    """
    Preprocess image for inference
    
    Args:
        image_path: Path to image
        size: Image size (default: 256)
    
    Returns:
        image_tensor: Preprocessed image tensor
        original_image: Original image for visualization
    """
    # Read image
    original_image = cv2.imread(str(image_path))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(original_image, (size, size))
    image = image.astype(np.float32) / 255.0
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor, original_image


def infer(model, image_tensor, device='cuda'):
    """
    Run inference
    
    Args:
        model: PCRN model
        image_tensor: Input image tensor
        device: Device to run on
    
    Returns:
        prediction: Binary segmentation mask
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        prediction = (output > 0.5).cpu().numpy().astype(np.uint8)
    
    return prediction.squeeze()


def visualize_results(original_image, prediction, output_path=None):
    """
    Visualize original image and prediction
    
    Args:
        original_image: Original image
        prediction: Predicted mask
        output_path: Path to save visualization (optional)
    """
    import matplotlib.pyplot as plt
    
    # Resize prediction to match original image
    h, w = original_image.shape[:2]
    prediction_resized = cv2.resize(prediction, (w, h))
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(prediction_resized, cmap='gray')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')
    
    # Overlay
    overlay = original_image.copy()
    overlay[prediction_resized > 0] = [255, 0, 0]
    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Visualization saved to {output_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = load_model(
        backbone_path='path/to/pvt_v2_b2.pth',
        model_path='path/to/model_checkpoint.pth',
        device=device
    )
    
    # Run inference
    image_path = 'path/to/image.jpg'
    image_tensor, original_image = preprocess_image(image_path)
    prediction = infer(model, image_tensor, device=device)
    
    # Visualize
    visualize_results(original_image, prediction, output_path='output.png')
