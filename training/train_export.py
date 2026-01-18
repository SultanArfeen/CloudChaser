"""
CloudChaser Model Training & ONNX Export - ENHANCED VERSION
Uses MobileNetV3-Large with stronger augmentation and longer training
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import MobileNet_V3_Large_Weights, EfficientNet_B0_Weights
from PIL import Image
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from collections import Counter
import random
import math


# Cloud class names
CLASS_NAMES = [
    "Cirriform",       # 0 - High altitude, ice crystals
    "Cumuliform",      # 1 - Vertically developed, dense
    "Stratiform",      # 2 - Layered, uniform
    "Stratocumuliform", # 3 - Hybrid rolling masses
    "Background"       # 4 - Trees, buildings, clear sky
]

# LWC (Liquid Water Content) ranges in g/m³
LWC_RANGES = {
    0: (0.01, 0.05),
    1: (0.5, 3.0),
    2: (0.25, 0.30),
    3: (0.30, 0.45),
    4: (0.0, 0.0),
}


class CloudDataset(Dataset):
    """Dataset for cloud classification from YOLO-format labels"""
    
    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        transform=None,
        max_samples: Optional[int] = None
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        
        # Find all images with labels
        self.samples = []
        self.class_counts = Counter()
        
        for img_path in self.images_dir.glob("*.*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                label_path = self.labels_dir / (img_path.stem + '.txt')
                if label_path.exists():
                    # Get majority class
                    classes = []
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                classes.append(int(parts[0]))
                    
                    label = Counter(classes).most_common(1)[0][0] if classes else 4
                    self.samples.append((img_path, label_path, label))
                    self.class_counts[label] += 1
        
        if max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} samples from {images_dir}")
        print(f"Class distribution: {dict(self.class_counts)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced dataset"""
        total = sum(self.class_counts.values())
        weights = []
        for i in range(5):
            count = self.class_counts.get(i, 1)
            weights.append(total / (5 * count))
        return torch.FloatTensor(weights)
    
    def get_sample_weights(self):
        """Get per-sample weights for WeightedRandomSampler"""
        class_weights = self.get_class_weights()
        return [class_weights[label] for _, _, label in self.samples]


class CloudClassifier(nn.Module):
    """Enhanced Cloud Classifier with multiple backbone options"""
    
    def __init__(self, num_classes: int = 5, backbone: str = 'mobilenet_large', dropout: float = 0.3):
        super().__init__()
        
        self.backbone_name = backbone
        
        if backbone == 'mobilenet_large':
            # MobileNetV3-Large - more parameters, better accuracy
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
            self.backbone = models.mobilenet_v3_large(weights=weights)
            in_features = self.backbone.classifier[0].in_features
            
            self.backbone.classifier = nn.Sequential(
                nn.Linear(in_features, 1280),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(1280, 640),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(640, num_classes)
            )
            
        elif backbone == 'efficientnet':
            # EfficientNet-B0 - state-of-the-art efficiency
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_features, 512),
                nn.SiLU(inplace=True),
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(512, num_classes)
            )
        
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
    
    def forward(self, x):
        return self.backbone(x)


def get_transforms(train: bool = True, img_size: int = 224):
    """Enhanced transforms with stronger augmentation"""
    
    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 48, img_size + 48)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler
) -> Tuple[float, float]:
    """Train for one epoch with mixed precision"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.1f}%")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, Dict]:
    """Validate model with per-class accuracy"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    class_correct = {i: 0 for i in range(5)}
    class_total = {i: 0 for i in range(5)}
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    per_class = {}
    for i in range(5):
        if class_total[i] > 0:
            per_class[CLASS_NAMES[i]] = 100.0 * class_correct[i] / class_total[i]
    
    return avg_loss, accuracy, per_class


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    img_size: int = 224,
    opset_version: int = 14
):
    """Export model to ONNX format"""
    model.eval()
    model.cpu()
    
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        do_constant_folding=True
    )
    
    print(f"\nModel exported to ONNX: {output_path}")
    print(f"  Opset version: {opset_version}")
    print(f"  Input shape: (batch, 3, {img_size}, {img_size})")
    
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("  ONNX verification: PASSED ✓")
    except Exception as e:
        print(f"  ONNX verification: {e}")


def train_model(
    dataset_path: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    img_size: int = 224,
    backbone: str = 'mobilenet_large',
    max_samples: Optional[int] = None
):
    """Full training pipeline with enhancements"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print("CloudChaser Enhanced Training")
    print("="*60)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"Backbone: {backbone}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("="*60)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup datasets
    train_dataset = CloudDataset(
        dataset_path / "images" / "train",
        dataset_path / "labels" / "train",
        transform=get_transforms(train=True, img_size=img_size),
        max_samples=max_samples
    )
    
    val_dataset = CloudDataset(
        dataset_path / "images" / "val",
        dataset_path / "labels" / "val",
        transform=get_transforms(train=False, img_size=img_size),
        max_samples=max_samples // 4 if max_samples else None
    )
    
    # Weighted sampler for class imbalance
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = CloudClassifier(num_classes=5, backbone=backbone, dropout=0.3)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Loss with class weights
    class_weights = train_dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Scheduler with warmup
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=5, total_epochs=epochs)
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler()
    
    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        val_loss, val_acc, per_class = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Per-class accuracy:")
        for cls_name, acc in per_class.items():
            print(f"  {cls_name}: {acc:.1f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'backbone': backbone,
            }, output_dir / 'cloudchaser_best.pth')
            print(f"  ★ New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    # Load best model for export
    best_checkpoint = torch.load(output_dir / 'cloudchaser_best.pth')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'val_acc': best_val_acc,
        'history': history,
        'class_names': CLASS_NAMES,
        'backbone': backbone,
    }, output_dir / 'cloudchaser_final.pth')
    
    # Export to ONNX
    print("\n" + "="*60)
    print("Exporting best model to ONNX format...")
    export_to_onnx(
        model,
        output_dir / 'cloudchaser.onnx',
        img_size=img_size,
        opset_version=14
    )
    
    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to: {output_dir}")
    
    return model, history


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CloudChaser model (Enhanced)")
    parser.add_argument('--dataset', type=str, default='./yolo_dataset',
                        help='Path to YOLO dataset')
    parser.add_argument('--output', type=str, default='./models',
                        help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--backbone', type=str, default='mobilenet_large',
                        choices=['mobilenet_large', 'efficientnet'],
                        help='Model backbone')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples (for testing)')
    parser.add_argument('--export-only', type=str, default=None,
                        help='Path to .pth file to export to ONNX')
    
    args = parser.parse_args()
    
    if args.export_only:
        print(f"Loading model from {args.export_only}")
        checkpoint = torch.load(args.export_only, map_location='cpu')
        backbone = checkpoint.get('backbone', 'mobilenet_large')
        model = CloudClassifier(num_classes=5, backbone=backbone)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        output_path = Path(args.export_only).with_suffix('.onnx')
        export_to_onnx(model, output_path, args.img_size)
    else:
        train_model(
            Path(args.dataset),
            Path(args.output),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            img_size=args.img_size,
            backbone=args.backbone,
            max_samples=args.max_samples
        )


if __name__ == "__main__":
    main()
