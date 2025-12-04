"""
Grad-CAM Visualization for Cataract Detection Model
Highlights which parts of the eye image influence the model's prediction
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from model import create_model
import os


class GradCAM:
    """Grad-CAM implementation for visualizing model attention"""
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model
            target_layer: Target layer for Grad-CAM (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save activations from forward pass"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save gradients from backward pass"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_image: Preprocessed input tensor (1, 3, H, W)
            target_class: Target class index (if None, uses predicted class)
        
        Returns:
            cam: Grad-CAM heatmap (H, W)
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        # Get predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients (weights)
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy(), target_class


def load_model(model_path='pytorch_checkpoints/best_model.pth'):
    """Load trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_model(num_classes=2, dropout_rate=0.5, freeze_backbone=False)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device


def get_target_layer(model):
    """Get the last convolutional layer from EfficientNet"""
    # For EfficientNet_B0, the last conv layer is in features
    # Navigate through the model structure
    for name, module in model.named_modules():
        if 'features' in name and isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv


def preprocess_image(image_path):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    input_tensor = transform(image).unsqueeze(0)
    
    return input_tensor, original_image


def overlay_heatmap(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on original image
    
    Args:
        image: Original image (H, W, 3) in RGB
        heatmap: Grad-CAM heatmap (H, W) in range [0, 1]
        alpha: Transparency of heatmap overlay
        colormap: OpenCV colormap
    
    Returns:
        overlay: Image with heatmap overlay
    """
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to colormap
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on image
    overlay = cv2.addWeighted(
        image, 1 - alpha,
        heatmap_colored, alpha,
        0
    )
    
    return overlay


def visualize_gradcam(image_path, model_path='pytorch_checkpoints/best_model.pth', 
                      save_path=None, show=True):
    """
    Generate and visualize Grad-CAM for an image
    
    Args:
        image_path: Path to input image
        model_path: Path to trained model
        save_path: Path to save visualization (if None, auto-generated)
        show: Whether to display the visualization
    
    Returns:
        prediction: Model prediction (class and probability)
        cam: Grad-CAM heatmap
    """
    # Load model
    model, device = load_model(model_path)
    
    # Get target layer (last conv layer)
    target_layer = get_target_layer(model)
    print(f"Target layer for Grad-CAM: {target_layer}")
    
    # Create Grad-CAM object
    gradcam = GradCAM(model, target_layer)
    
    # Preprocess image
    input_tensor, original_image = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[predicted_class].item()
    
    class_names = ['Cataract', 'Normal']
    prediction_text = f"{class_names[predicted_class]} ({confidence*100:.2f}%)"
    
    # Generate Grad-CAM
    cam, target_class = gradcam.generate_cam(input_tensor, target_class=predicted_class)
    
    # Create overlay
    overlay = overlay_heatmap(original_image, cam, alpha=0.5)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Grad-CAM heatmap
    im = axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay - Prediction: {prediction_text}', 
                     fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    if save_path is None:
        filename = os.path.basename(image_path)
        save_path = f'gradcam_{filename}'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Grad-CAM visualization saved: {save_path}")
    
    # Show plot
    if show:
        plt.show()
    else:
        plt.close()
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Prediction: {class_names[predicted_class]}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"Cataract probability: {probabilities[0].item()*100:.2f}%")
    print(f"Normal probability: {probabilities[1].item()*100:.2f}%")
    print(f"{'='*60}\n")
    
    return {
        'prediction': class_names[predicted_class],
        'confidence': confidence,
        'probabilities': probabilities.cpu().numpy(),
        'cam': cam,
        'overlay': overlay
    }


def batch_visualize(image_paths, model_path='pytorch_checkpoints/best_model.pth',
                   output_dir='gradcam_results'):
    """
    Generate Grad-CAM visualizations for multiple images
    
    Args:
        image_paths: List of image paths
        model_path: Path to trained model
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating Grad-CAM for {len(image_paths)} images...")
    print(f"{'='*60}\n")
    
    results = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Processing: {os.path.basename(image_path)}")
        
        filename = os.path.basename(image_path)
        save_path = os.path.join(output_dir, f'gradcam_{filename}')
        
        result = visualize_gradcam(
            image_path, 
            model_path=model_path,
            save_path=save_path,
            show=False
        )
        results.append({
            'image': filename,
            'prediction': result['prediction'],
            'confidence': result['confidence']
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"{r['image']}: {r['prediction']} ({r['confidence']*100:.2f}%)")
    print(f"{'='*60}\n")
    print(f"✅ All visualizations saved in: {output_dir}/")


if __name__ == '__main__':
    # Example usage
    print("\n" + "="*60)
    print("GRAD-CAM VISUALIZATION FOR CATARACT DETECTION")
    print("="*60 + "\n")
    
    # Test on user's screenshot (normal eye)
    print("Testing on normal eye screenshot...")
    test_image = r"C:\Users\Harshita\Downloads\Kartik cataract\cataract detection\test_images\kartik_phone_eye.jpg"
    
    if os.path.exists(test_image):
        result = visualize_gradcam(test_image, show=True)
    else:
        print(f"⚠️ Test image not found: {test_image}")
        print("\nUsage:")
        print("  python gradcam_visualization.py")
        print("\nOr in Python:")
        print("  from gradcam_visualization import visualize_gradcam")
        print("  result = visualize_gradcam('path/to/image.jpg')")
        print("\nFor batch processing:")
        print("  from gradcam_visualization import batch_visualize")
        print("  batch_visualize(['image1.jpg', 'image2.jpg', ...])")
