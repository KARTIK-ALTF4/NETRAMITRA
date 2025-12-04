"""
PyTorch DataLoader for Binary Image Classification (Cataract vs Normal)
Loads images from dataset/train and dataset/val folders
Applies resizing (224x224), normalization, and augmentation
Returns batches of tensors ready for transfer learning
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path


class CataractDataset(Dataset):
    """
    Custom Dataset for Cataract Detection
    Expects folder structure:
        dataset/
            train/
                cataract/
                    img1.jpg, img2.jpg, ...
                normal/
                    img1.jpg, img2.jpg, ...
            val/
                cataract/
                normal/
    """
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Root directory path (e.g., 'dataset/train' or 'dataset/val')
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = ['cataract', 'normal']
        self.class_to_idx = {'cataract': 0, 'normal': 1}
        
        # Load all image paths and labels
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Directory {class_dir} does not exist!")
                continue
            
            # Get all image files
            image_files = list(class_dir.glob('*.jpg')) + \
                         list(class_dir.glob('*.jpeg')) + \
                         list(class_dir.glob('*.png'))
            
            for img_path in image_files:
                self.images.append(str(img_path))
                self.labels.append(self.class_to_idx[class_name])
        
        print(f"Loaded {len(self.images)} images from {root_dir}")
        print(f"  - Cataract: {self.labels.count(0)}")
        print(f"  - Normal: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        
        Returns:
            tuple: (image, label) where image is a tensor and label is an integer
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(train=True):
    """
    Get image transforms for training or validation
    Realistic augmentations for eye-image photos that preserve medical features
    
    Args:
        train (bool): If True, apply data augmentation. If False, only resize and normalize.
    
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    if train:
        # Training transforms with realistic eye-image augmentation
        # Designed to preserve medical features (lens opacity, iris texture, pupil clarity)
        return transforms.Compose([
            transforms.Resize((224, 224)),
            
            # Rotation: ±15° (eyes can be photographed at slight angles)
            transforms.RandomRotation(degrees=15),
            
            # Horizontal flip: 50% chance (left/right eye doesn't affect diagnosis)
            transforms.RandomHorizontalFlip(p=0.5),
            
            # Brightness: ±30% (different lighting conditions, flash vs no flash)
            # Contrast: ±20% (camera quality variations)
            # Saturation: ±20% (different camera color profiles)
            # Hue: ±5% (very mild, preserve natural eye color)
            transforms.ColorJitter(
                brightness=0.3,  # Range: 0.7x to 1.3x
                contrast=0.2,    # Range: 0.8x to 1.2x
                saturation=0.2,  # Range: 0.8x to 1.2x
                hue=0.05         # Range: -0.05 to +0.05 (mild)
            ),
            
            # Random affine: slight zoom and translation
            # Simulates different camera distances and framing
            transforms.RandomAffine(
                degrees=0,           # No extra rotation (already handled above)
                translate=(0.05, 0.05),  # ±5% shift in x and y
                scale=(0.95, 1.05),      # 95% to 105% zoom
                shear=0              # No shear (would distort eye shape)
            ),
            
            # Gaussian blur: 10% chance, mild blur (simulates slight out-of-focus)
            # Kernel size 3 is very mild, preserves medical details
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            ], p=0.1),
            
            # Convert to tensor
            transforms.ToTensor(),
            
            # ImageNet normalization (standard for transfer learning)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]     # ImageNet std
            )
        ])
    else:
        # Validation/Test transforms (no augmentation)
        # Only basic preprocessing to match model input requirements
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def get_dataloaders(data_dir='dataset', batch_size=32, num_workers=4):
    """
    Create DataLoaders for training and validation
    
    Args:
        data_dir (str): Root directory containing 'train' and 'val' folders
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of worker processes for data loading
    
    Returns:
        tuple: (train_loader, val_loader, class_names)
    """
    data_dir = Path(data_dir)
    
    # Create datasets
    train_dataset = CataractDataset(
        root_dir=data_dir / 'train',
        transform=get_transforms(train=True)
    )
    
    val_dataset = CataractDataset(
        root_dir=data_dir / 'val',
        transform=get_transforms(train=False)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    class_names = ['cataract', 'normal']
    
    print("\n" + "="*60)
    print("DataLoaders Created Successfully!")
    print("="*60)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Batch size: {batch_size}")
    print(f"Classes: {class_names}")
    print("="*60 + "\n")
    
    return train_loader, val_loader, class_names


# Example usage
if __name__ == "__main__":
    """
    Test the dataloader
    """
    print("Testing PyTorch DataLoader for Cataract Detection\n")
    
    # Check if dataset exists
    if not os.path.exists('dataset/train'):
        print("❌ 'dataset/train' folder not found!")
        print("Please create the folder structure:")
        print("  dataset/")
        print("    train/")
        print("      cataract/")
        print("      normal/")
        print("    val/")
        print("      cataract/")
        print("      normal/")
    else:
        # Create dataloaders
        train_loader, val_loader, class_names = get_dataloaders(
            data_dir='dataset',
            batch_size=32,
            num_workers=0  # Use 0 for Windows, 4+ for Linux/Mac
        )
        
        # Test: Load one batch
        print("Loading one batch from training set...")
        for images, labels in train_loader:
            print(f"✅ Batch shape: {images.shape}")  # Should be [batch_size, 3, 224, 224]
            print(f"✅ Labels shape: {labels.shape}")  # Should be [batch_size]
            print(f"✅ Image dtype: {images.dtype}")  # Should be torch.float32
            print(f"✅ Label values: {labels[:8].tolist()}")  # First 8 labels
            print(f"✅ Min pixel value: {images.min():.3f}")
            print(f"✅ Max pixel value: {images.max():.3f}")
            break
        
        print("\n✅ DataLoader is working correctly!")
