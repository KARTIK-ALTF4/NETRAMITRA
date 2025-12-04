"""
Evaluate the trained PyTorch model on test set
Generates accuracy, precision, recall, F1-score and confusion matrix
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import our custom modules
from model import create_model
from dataloader import get_dataloaders

def evaluate_model(model_path, data_dir, batch_size=32):
    """
    Evaluate model on test set
    
    Args:
        model_path: Path to saved model checkpoint
        data_dir: Directory containing train/validation/test folders
        batch_size: Batch size for evaluation
    """
    
    print("="*70)
    print(" CATARACT DETECTION MODEL EVALUATION ")
    print("="*70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✅ Using device: {device}")
    
    # Load model
    print("\n📦 Loading model...")
    model = create_model(num_classes=2, dropout_rate=0.5, freeze_backbone=False)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"   Loaded checkpoint from epoch {checkpoint.get('epoch', 'Unknown')}")
    print(f"   Validation accuracy: {checkpoint.get('val_acc', 'Unknown'):.4f}")
    print(f"   Validation loss: {checkpoint.get('val_loss', 'Unknown'):.6f}")
    
    # Load test data
    print("\n📊 Loading test data...")
    # Get dataloaders (we only need test but function returns train/val too)
    try:
        from torchvision import datasets, transforms
        
        # Define the same transforms used during training
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load test dataset
        test_dir = Path(data_dir) / 'test'
        test_dataset = datasets.ImageFolder(str(test_dir), transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        print(f"   Test samples: {len(test_dataset)}")
        print(f"   Classes: {test_dataset.classes}")
        
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return
    
    # Evaluate
    print("\n🔍 Evaluating on test set...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    print("\n" + "="*70)
    print(" TEST SET RESULTS ")
    print("="*70)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', pos_label=1)
    recall = recall_score(all_labels, all_preds, average='binary', pos_label=1)
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1)
    
    print(f"\n📈 Overall Metrics:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Per-class metrics
    print(f"\n📊 Per-Class Metrics:")
    for class_idx, class_name in enumerate(test_dataset.classes):
        class_mask = all_labels == class_idx
        class_accuracy = accuracy_score(all_labels[class_mask], all_preds[class_mask])
        print(f"   {class_name}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\n🔢 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"               Cataract  Normal")
    print(f"   Actual")
    print(f"   Cataract     {cm[0][0]:4d}     {cm[0][1]:4d}")
    print(f"   Normal       {cm[1][0]:4d}     {cm[1][1]:4d}")
    
    # Calculate additional stats
    tn, fp, fn, tp = cm[1][1], cm[1][0], cm[0][1], cm[0][0]
    
    print(f"\n📉 Detailed Classification:")
    print(f"   True Positives (Cataract correctly identified):  {tp}")
    print(f"   True Negatives (Normal correctly identified):    {tn}")
    print(f"   False Positives (Normal misclassified as Cataract): {fp}")
    print(f"   False Negatives (Cataract misclassified as Normal): {fn}")
    
    # Plot confusion matrix
    print("\n📊 Generating confusion matrix plot...")
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Cataract', 'Normal'],
                yticklabels=['Cataract', 'Normal'],
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Cataract Detection\nTest Set Evaluation', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    # Add accuracy text
    plt.text(0.5, -0.15, f'Overall Accuracy: {accuracy*100:.2f}%', 
             ha='center', transform=plt.gca().transAxes, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Confusion matrix saved to: {output_path}")
    
    # Show plot
    plt.show()
    
    print("\n" + "="*70)
    print(" EVALUATION COMPLETE! ")
    print("="*70)


if __name__ == "__main__":
    # Configuration
    model_path = 'pytorch_checkpoints/best_model.pth'
    data_dir = 'comprehensive_data'  # Directory with train/validation/test folders
    batch_size = 32
    
    # Run evaluation
    evaluate_model(model_path, data_dir, batch_size)
