
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class CataractClassifier(nn.Module):
    """
    Transfer Learning Model for Cataract Detection
    Base: EfficientNet_B0 (pretrained on ImageNet)
    Task: Binary classification (Cataract=0, Normal=1)
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.5, freeze_backbone=False):
        """
        Args:
            num_classes (int): Number of output classes (default: 2 for binary classification)
            dropout_rate (float): Dropout probability for regularization (default: 0.5)
            freeze_backbone (bool): If True, freeze all backbone layers (default: False)
        """
        super(CataractClassifier, self).__init__()
        
        # Load pretrained EfficientNet_B0
        print("Loading EfficientNet_B0 pretrained on ImageNet...")
        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Get the number of input features for the classifier
        in_features = self.backbone.classifier[1].in_features
        
        # Replace the final classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),  # Dropout for regularization
            nn.Linear(in_features, 512),               # Dense layer 1
            nn.ReLU(inplace=True),                     # Activation
            nn.Dropout(p=dropout_rate * 0.5),          # Mild dropout (0.25 if dropout_rate=0.5)
            nn.Linear(512, num_classes)                # Output layer (2 classes)
        )
        
        # Optionally freeze backbone layers
        if freeze_backbone:
            print("Freezing backbone layers (only training classifier head)...")
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        else:
            print("Backbone layers are trainable (fine-tuning entire network)...")
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        print(f"✅ Model created:")
        print(f"   - Backbone: EfficientNet_B0")
        print(f"   - Output classes: {num_classes}")
        print(f"   - Dropout rate: {dropout_rate}")
        print(f"   - Backbone frozen: {freeze_backbone}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, 224, 224]
        
        Returns:
            torch.Tensor: Output logits of shape [batch_size, num_classes]
        """
        return self.backbone(x)
    
    def unfreeze_backbone(self, num_layers=None):
        """
        Unfreeze backbone layers for fine-tuning
        
        Args:
            num_layers (int, optional): Number of layers to unfreeze from the end.
                                       If None, unfreeze all layers.
        """
        if num_layers is None:
            print("Unfreezing all backbone layers...")
            for param in self.backbone.features.parameters():
                param.requires_grad = True
        else:
            print(f"Unfreezing last {num_layers} backbone layers...")
            layers = list(self.backbone.features.children())
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
    
    def get_trainable_params(self):
        """
        Get number of trainable parameters
        
        Returns:
            tuple: (trainable_params, total_params)
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def create_model(num_classes=2, dropout_rate=0.5, freeze_backbone=False, device='cuda'):
    """
    Factory function to create and initialize the model
    
    Args:
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout probability
        freeze_backbone (bool): Whether to freeze backbone layers
        device (str): Device to move model to ('cuda' or 'cpu')
    
    Returns:
        CataractClassifier: Initialized model
    """
    model = CataractClassifier(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone
    )
    
    # Move to device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Print model summary
    trainable, total = model.get_trainable_params()
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Non-trainable parameters: {total - trainable:,}")
    print(f"Device: {device}")
    print("="*60 + "\n")
    
    return model


# Example usage and testing
if __name__ == "__main__":
    print("Testing EfficientNet_B0 Transfer Learning Model\n")
    
    # Create model
    model = create_model(
        num_classes=2,
        dropout_rate=0.5,
        freeze_backbone=False,  # Fine-tune entire network
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Test forward pass
    print("Testing forward pass...")
    device = next(model.parameters()).device
    dummy_input = torch.randn(4, 3, 224, 224).to(device)  # Batch of 4 images
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✅ Input shape: {dummy_input.shape}")
    print(f"✅ Output shape: {output.shape}")  # Should be [4, 2]
    print(f"✅ Output logits (first sample): {output[0].cpu().numpy()}")
    
    # Test with softmax (probabilities)
    probs = torch.softmax(output, dim=1)
    print(f"✅ Probabilities (first sample): {probs[0].cpu().numpy()}")
    print(f"   - Cataract probability: {probs[0, 0].item():.4f}")
    print(f"   - Normal probability: {probs[0, 1].item():.4f}")
    
    print("\n✅ Model is working correctly!")
    
    # Example: Freeze backbone and check trainable params
    print("\n" + "-"*60)
    print("Testing freeze/unfreeze functionality...")
    print("-"*60)
    
    model2 = create_model(num_classes=2, freeze_backbone=True, device='cpu')
    trainable_frozen, total_frozen = model2.get_trainable_params()
    print(f"With frozen backbone: {trainable_frozen:,} trainable params")
    
    model2.unfreeze_backbone()
    trainable_unfrozen, total_unfrozen = model2.get_trainable_params()
    print(f"After unfreezing: {trainable_unfrozen:,} trainable params")
