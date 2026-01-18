"""
CloudChaser ETL Pipeline
Converts Clouds-1500 Supervisely JSON annotations to YOLO format
"""

import os
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class BoundingBox:
    """YOLO format bounding box (normalized coordinates)"""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    
    def to_yolo_line(self) -> str:
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


# Class mapping (Supervisely Portuguese -> YOLO class ID)
CLASS_MAPPING = {
    "Cirriformes": 0,      # High-altitude ice crystals
    "Cumuliformes": 1,     # Vertically developed, dense
    "Estratiformes": 2,    # Layered, uniform (Stratiform)
    "Estratocumuliformes": 3,  # Hybrid rolling masses (Stratocumuliform)
    "Arvore": 4,           # Background (trees, buildings)
}

# Reverse mapping for display
CLASS_NAMES = {v: k for k, v in CLASS_MAPPING.items()}


def polygon_to_bbox(points: List[List[float]], img_width: int, img_height: int) -> BoundingBox | None:
    """
    Convert Supervisely polygon exterior points to normalized YOLO bounding box.
    
    Args:
        points: List of [x, y] coordinate pairs
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        BoundingBox with normalized coordinates, or None if invalid
    """
    if not points or len(points) < 3:
        return None
    
    # Extract x and y coordinates
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    # Calculate bounding box extremes
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Calculate absolute dimensions
    w_abs = x_max - x_min
    h_abs = y_max - y_min
    
    # Skip tiny or invalid boxes
    if w_abs < 5 or h_abs < 5:
        return None
    
    # Calculate center
    x_center_abs = x_min + w_abs / 2
    y_center_abs = y_min + h_abs / 2
    
    # Normalize to [0, 1]
    x_center = x_center_abs / img_width
    y_center = y_center_abs / img_height
    width = w_abs / img_width
    height = h_abs / img_height
    
    # Clamp to valid range
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return BoundingBox(
        class_id=-1,  # Will be set later
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height
    )


def parse_supervisely_annotation(json_path: Path, meta_classes: Dict[int, str]) -> Tuple[List[BoundingBox], int, int]:
    """
    Parse a Supervisely JSON annotation file.
    
    Args:
        json_path: Path to the .json annotation file
        meta_classes: Mapping of classId -> className from meta.json
    
    Returns:
        Tuple of (list of bounding boxes, image width, image height)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get image dimensions from the annotation
    img_width = data.get('size', {}).get('width', 2592)  # Default from dataset
    img_height = data.get('size', {}).get('height', 1944)
    
    boxes = []
    
    for obj in data.get('objects', []):
        # Get class name
        class_title = obj.get('classTitle', '')
        if class_title not in CLASS_MAPPING:
            continue
        
        class_id = CLASS_MAPPING[class_title]
        
        # Get polygon points
        points = obj.get('points', {}).get('exterior', [])
        if not points:
            continue
        
        # Convert to bounding box
        bbox = polygon_to_bbox(points, img_width, img_height)
        if bbox:
            bbox.class_id = class_id
            boxes.append(bbox)
    
    return boxes, img_width, img_height


def load_meta_json(dataset_root: Path) -> Dict[int, str]:
    """Load class definitions from meta.json"""
    meta_path = dataset_root / "meta.json"
    if not meta_path.exists():
        return {}
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    class_map = {}
    for cls in meta.get('classes', []):
        class_id = cls.get('id')
        class_title = cls.get('title')
        if class_id and class_title:
            class_map[class_id] = class_title
    
    return class_map


def find_annotation_folders(dataset_root: Path) -> List[Tuple[Path, Path]]:
    """
    Recursively find all annotation folders and their corresponding image folders.
    
    Returns:
        List of (ann_folder, img_folder) tuples
    """
    pairs = []
    
    for root, dirs, files in os.walk(dataset_root):
        root_path = Path(root)
        
        # Check if this directory has 'ann' and 'img' subdirectories
        ann_path = root_path / 'ann'
        img_path = root_path / 'img'
        
        if ann_path.is_dir() and img_path.is_dir():
            pairs.append((ann_path, img_path))
    
    return pairs


def process_dataset(
    dataset_root: Path,
    output_root: Path,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Process entire Clouds-1500 dataset and create YOLO format output.
    
    Args:
        dataset_root: Root path to CCDataset
        output_root: Output path for YOLO dataset
        train_ratio: Ratio for train/val split (default 80/20)
        seed: Random seed for reproducibility
    
    Returns:
        Statistics dictionary
    """
    random.seed(seed)
    
    # Create output directories
    train_images = output_root / "images" / "train"
    train_labels = output_root / "labels" / "train"
    val_images = output_root / "images" / "val"
    val_labels = output_root / "labels" / "val"
    
    for d in [train_images, train_labels, val_images, val_labels]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Load meta classes
    meta_classes = load_meta_json(dataset_root)
    print(f"Loaded {len(meta_classes)} classes from meta.json")
    
    # Find all annotation folders
    folder_pairs = find_annotation_folders(dataset_root)
    print(f"Found {len(folder_pairs)} annotation folders")
    
    # Collect all samples
    samples = []  # List of (ann_path, img_path, relative_name)
    
    for ann_folder, img_folder in folder_pairs:
        for ann_file in ann_folder.glob("*.json"):
            # Get corresponding image name (remove .json extension)
            img_name = ann_file.stem  # e.g., "08-57-00.jpg"
            
            # Look for image file
            img_path = img_folder / img_name
            if img_path.exists():
                # Create unique name from folder structure
                folder_name = ann_folder.parent.name.replace(" ", "_").replace("-", "_")
                unique_name = f"{folder_name}_{img_name}"
                samples.append((ann_file, img_path, unique_name))
    
    print(f"Found {len(samples)} valid image-annotation pairs")
    
    # Count classes for stratified split
    class_counts = {i: [] for i in range(5)}  # class_id -> list of sample indices
    
    for idx, (ann_path, _, _) in enumerate(samples):
        boxes, _, _ = parse_supervisely_annotation(ann_path, meta_classes)
        # Assign sample to its primary class (most common box class)
        if boxes:
            class_id = max(set(b.class_id for b in boxes), key=lambda c: sum(1 for b in boxes if b.class_id == c))
            class_counts[class_id].append(idx)
        else:
            class_counts[4].append(idx)  # Background
    
    # Stratified split
    train_indices = set()
    val_indices = set()
    
    for class_id, indices in class_counts.items():
        random.shuffle(indices)
        split_point = int(len(indices) * train_ratio)
        train_indices.update(indices[:split_point])
        val_indices.update(indices[split_point:])
    
    print(f"Split: {len(train_indices)} train, {len(val_indices)} val")
    
    # Process samples
    stats = {
        'total_images': len(samples),
        'train_images': len(train_indices),
        'val_images': len(val_indices),
        'total_boxes': 0,
        'class_distribution': {i: 0 for i in range(5)},
        'skipped_no_boxes': 0,
    }
    
    for idx, (ann_path, img_path, unique_name) in enumerate(samples):
        # Determine split
        is_train = idx in train_indices
        
        # Parse annotation
        boxes, img_w, img_h = parse_supervisely_annotation(ann_path, meta_classes)
        
        if not boxes:
            stats['skipped_no_boxes'] += 1
            continue
        
        # Choose output directories
        out_img_dir = train_images if is_train else val_images
        out_lbl_dir = train_labels if is_train else val_labels
        
        # Copy image
        out_img_path = out_img_dir / unique_name
        shutil.copy2(img_path, out_img_path)
        
        # Write YOLO label file
        label_name = unique_name.rsplit('.', 1)[0] + '.txt'
        out_lbl_path = out_lbl_dir / label_name
        
        with open(out_lbl_path, 'w') as f:
            for box in boxes:
                f.write(box.to_yolo_line() + '\n')
                stats['total_boxes'] += 1
                stats['class_distribution'][box.class_id] += 1
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(samples)} images...")
    
    # Create dataset YAML for YOLO training
    yaml_content = f"""# CloudChaser Dataset - Clouds-1500
path: {output_root.absolute()}
train: images/train
val: images/val

nc: 5
names:
  0: Cirriform
  1: Cumuliform  
  2: Stratiform
  3: Stratocumuliform
  4: Background
"""
    
    yaml_path = output_root / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print("\n" + "="*50)
    print("ETL COMPLETE")
    print("="*50)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Training images: {stats['train_images']}")
    print(f"Validation images: {stats['val_images']}")
    print(f"Skipped (no boxes): {stats['skipped_no_boxes']}")
    print(f"Total bounding boxes: {stats['total_boxes']}")
    print("\nClass distribution:")
    for class_id, count in stats['class_distribution'].items():
        print(f"  {CLASS_NAMES.get(class_id, 'Unknown')} ({class_id}): {count}")
    print(f"\nDataset YAML saved to: {yaml_path}")
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Clouds-1500 to YOLO format")
    parser.add_argument('--input', type=str, default='./CCDataset',
                        help='Path to CCDataset root')
    parser.add_argument('--output', type=str, default='./yolo_dataset',
                        help='Output path for YOLO dataset')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Training set ratio (default: 0.8)')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode (process only first folder)')
    
    args = parser.parse_args()
    
    dataset_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    
    if not dataset_root.exists():
        print(f"Error: Dataset not found at {dataset_root}")
        return 1
    
    print(f"Input: {dataset_root}")
    print(f"Output: {output_root}")
    print(f"Train ratio: {args.train_ratio}")
    
    if args.test:
        print("\n[TEST MODE] Processing limited data...")
    
    stats = process_dataset(dataset_root, output_root, args.train_ratio)
    
    return 0


if __name__ == "__main__":
    exit(main())
