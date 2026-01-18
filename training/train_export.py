"""
CloudChaser Model Training & ONNX Export
MobileNetV3-Small for cloud classification
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import MobileNet_V3_Small_Weights
from PIL import Image
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from collections import Counter
import random


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
    0: (0.01, 0.05),   # Cirriform - ice crystals, very low
    1: (0.5, 3.0),     # Cumuliform - high convective
    2: (0.25, 0.30),   # Stratiform - moderate, steady
    3: (0.30, 0.45),   # Stratocumuliform - intermediate
    4: (0.0, 0.0),     # Background - no cloud
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
        
        for img_path in self.images_dir.glob("*.*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # Find corresponding label
                label_path = self.labels_dir / (img_path.stem + '.txt')
                if label_path.exists():
                    self.samples.append((img_path, label_path))
        
        if max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} samples from {images_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load label - get majority class from bounding boxes
        classes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.append(int(parts[0]))
        
        # Use majority class, or 4 (background) if no boxes
        if classes:
            label = Counter(classes).most_common(1)[0][0]
        else:
            label = 4
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CloudClassifier(nn.Module):
    """MobileNetV3-Small based cloud classifier"""
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained MobileNetV3-Small
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self.backbone = models.mobilenet_v3_small(weights=weights)
        else:
            self.backbone = models.mobilenet_v3_small(weights=None)
        
        # Get the number of features from the classifier
        in_features = self.backbone.classifier[0].in_features
        
        # Replace classifier with custom head for 5 classes
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def get_transforms(train: bool = True, img_size: int = 224):
    """Get image transforms for training or inference"""
    
    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    img_size: int = 224,
    opset_version: int = 14  # CRITICAL: Must be 14+ for Hardsigmoid support
):
    """
    Export model to ONNX format for web inference.
    
    IMPORTANT: opset_version must be 14 or higher to support 
    Hardsigmoid and Hardswish activations used in MobileNetV3.
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    # Export
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
    
    print(f"Model exported to ONNX: {output_path}")
    print(f"  Opset version: {opset_version}")
    print(f"  Input shape: (batch, 3, {img_size}, {img_size})")
    
    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("  ONNX model verification: PASSED")
    except ImportError:
        print("  (Install 'onnx' package to verify exported model)")
    except Exception as e:
        print(f"  ONNX verification warning: {e}")


def train_model(
    dataset_path: Path,
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    img_size: int = 224,
    max_samples: Optional[int] = None
):
    """Full training pipeline"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
    model = CloudClassifier(num_classes=5, pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, output_dir / 'cloudchaser_best.pth')
            print(f"  -> New best model saved! (Val Acc: {val_acc:.2f}%)")
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'history': history,
        'class_names': CLASS_NAMES,
    }, output_dir / 'cloudchaser_final.pth')
    
    # Export to ONNX
    print("\n" + "="*50)
    print("Exporting to ONNX format...")
    export_to_onnx(
        model,
        output_dir / 'cloudchaser.onnx',
        img_size=img_size,
        opset_version=14
    )
    
    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to: {output_dir}")
    
    return model, history


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CloudChaser model")
    parser.add_argument('--dataset', type=str, default='./yolo_dataset',
                        help='Path to YOLO dataset')
    parser.add_argument('--output', type=str, default='./models',
                        help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples (for testing)')
    parser.add_argument('--export-only', type=str, default=None,
                        help='Path to .pth file to export to ONNX')
    
    args = parser.parse_args()
    
    if args.export_only:
        # Just export existing model
        print(f"Loading model from {args.export_only}")
        model = CloudClassifier(num_classes=5, pretrained=False)
        checkpoint = torch.load(args.export_only, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        output_path = Path(args.export_only).with_suffix('.onnx')
        export_to_onnx(model, output_path, args.img_size)
    else:
        # Full training
        train_model(
            Path(args.dataset),
            Path(args.output),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            img_size=args.img_size,
            max_samples=args.max_samples
        )


if __name__ == "__main__":
    main()
