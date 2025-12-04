"""
Complete Training Loop for Cataract Detection
Includes early stopping, validation accuracy tracking, and GPU support
Uses CrossEntropy loss and Adam optimizer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import copy
from pathlib import Path
import numpy as np


class EarlyStopping:
    """l
    Early stopping to stop training when validation loss doesn't improve
    """
    def __init__(self, patience=7, min_delta=0, verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as improvement
            verbose (bool): Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0


class MetricTracker:
    """
    Track training and validation metrics
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.accuracies = []
        self.all_preds = []
        self.all_labels = []
    
    def update(self, loss, preds, labels):
        self.losses.append(loss)
        self.all_preds.extend(preds.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
    
    def get_metrics(self):
        avg_loss = np.mean(self.losses)
        all_preds = np.array(self.all_preds)
        all_labels = np.array(self.all_labels)
        accuracy = (all_preds == all_labels).mean()
        return avg_loss, accuracy


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs):
    """
    Train for one epoch
    
    Args:
        model: PyTorch model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        num_epochs: Total number of epochs
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    tracker = MetricTracker()
    
    print(f"\nEpoch {epoch}/{num_epochs}")
    print("-" * 60)
    
    start_time = time.time()
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        # Move to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        _, preds = torch.max(outputs, 1)
        tracker.update(loss.item(), preds, labels)
        
        # Print progress
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            avg_loss, avg_acc = tracker.get_metrics()
            print(f"  Batch [{batch_idx + 1}/{len(dataloader)}] "
                  f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
    
    # Calculate final metrics
    avg_loss, avg_acc = tracker.get_metrics()
    epoch_time = time.time() - start_time
    
    print(f"  Time: {epoch_time:.2f}s | Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f}")
    
    return avg_loss, avg_acc


def validate_epoch(model, dataloader, criterion, device):
    """
    Validate the model
    
    Args:
        model: PyTorch model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    tracker = MetricTracker()
    
    with torch.no_grad():
        for images, labels in dataloader:
            # Move to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Track metrics
            _, preds = torch.max(outputs, 1)
            tracker.update(loss.item(), preds, labels)
    
    # Calculate metrics
    avg_loss, avg_acc = tracker.get_metrics()
    
    print(f"  Val Loss: {avg_loss:.4f} | Val Acc: {avg_acc:.4f}")
    
    return avg_loss, avg_acc


def train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=1e-4,
                patience=7, save_dir='checkpoints', log_dir='runs'):
    """
    Complete training loop with early stopping and validation tracking
    
    Args:
        model: PyTorch model to train
        train_loader: Training dataloader
        val_loader: Validation dataloader
        num_epochs (int): Maximum number of epochs
        learning_rate (float): Learning rate for Adam optimizer
        patience (int): Early stopping patience
        save_dir (str): Directory to save model checkpoints
        log_dir (str): Directory for tensorboard logs
    
    Returns:
        tuple: (trained_model, history_dict)
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Early stopping patience: {patience}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print("="*60)
    
    # Calculate class weights for imbalanced dataset
    # Count samples per class from train_loader
    class_counts = torch.zeros(2)
    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1
    
    # Compute weights (inverse frequency)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * 2  # Normalize
    class_weights = class_weights.to(device)
    
    print(f"\n📊 Class Distribution:")
    print(f"  Cataract (0): {int(class_counts[0])} samples, weight: {class_weights[0]:.3f}")
    print(f"  Normal (1):   {int(class_counts[1])} samples, weight: {class_weights[1]:.3f}")
    print(f"  Ratio: 1:{class_counts[1]/class_counts[0]:.1f}")
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler (reduce on plateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    # TensorBoard writer
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Create checkpoint directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # History tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    # Best model tracking
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')  # Changed to track loss instead of accuracy
    best_epoch = 0
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)
        
        # Save best model (based on validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, f'{save_dir}/best_model.pth')
            print(f"  ✅ Best model saved! (Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f})")
        
        # Early stopping check
        early_stopping(val_loss, epoch)
        if early_stopping.early_stop:
            print(f"\n⚠️ Early stopping triggered at epoch {epoch}")
            print(f"Best epoch was {early_stopping.best_epoch} with val_loss={early_stopping.best_loss:.4f}")
            break
        
        print("-" * 60)
    
    # Training complete
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Total time: {total_time / 60:.2f} minutes")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Best model saved to: {save_dir}/best_model.pth")
    print("="*60 + "\n")
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Close writer
    writer.close()
    
    return model, history


# Example usage
if __name__ == "__main__":
    print("Testing Training Loop\n")
    
    # This is just a test - you would normally import your model and dataloaders
    print("To use this training loop:")
    print("1. Import your model from model.py")
    print("2. Import your dataloaders from dataloader.py")
    print("3. Call train_model() with your configuration")
    print("\nExample:")
    print("```python")
    print("from model import create_model")
    print("from dataloader import get_dataloaders")
    print("from train import train_model")
    print("")
    print("# Create model")
    print("model = create_model(num_classes=2, dropout_rate=0.5)")
    print("")
    print("# Load data")
    print("train_loader, val_loader, _ = get_dataloaders(")
    print("    data_dir='dataset', batch_size=32")
    print(")")
    print("")
    print("# Train")
    print("trained_model, history = train_model(")
    print("    model, train_loader, val_loader,")
    print("    num_epochs=30, learning_rate=1e-4, patience=7")
    print(")")
    print("```")
