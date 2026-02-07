import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any


class CocoDataset(Dataset):
    """
    COCO Dataset for object detection.
    Loads images and annotations in COCO format.
    """
    
    def __init__(
        self, 
        root: str, 
        ann_file: str, 
        img_size: Tuple[int, int] = (224, 224),
        transform: Any = None
    ):
        """
        Args:
            root: Root directory of images
            ann_file: Path to COCO annotation JSON file
            img_size: Target image size (H, W)
            transform: Custom transforms (optional)
        """
        self.root = root
        self.img_size = img_size
        self.transform = transform
        
        # Load annotations
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
            
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: i for i, cat in enumerate(self.coco_data['categories'])}
        self.num_classes = len(self.categories)
        
        # Group annotations by image_id
        self.annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
            
        self.img_ids = sorted(list(self.images.keys()))
        
    def __len__(self) -> int:
        return len(self.img_ids)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.root, img_info['file_name'])
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        
        # Resize image
        img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        
        # Load annotations
        anns = self.annotations.get(img_id, [])
        boxes = []
        labels = []
        
        for ann in anns:
            # COCO bbox: [x, y, width, height]
            bbox = ann['bbox']
            x, y, w, h = bbox
            
            # Convert to xyxy and scale to img_size
            x1 = x * self.img_size[1] / orig_w
            y1 = y * self.img_size[0] / orig_h
            x2 = (x + w) * self.img_size[1] / orig_w
            y2 = (y + h) * self.img_size[0] / orig_h
            
            boxes.append([x1, y1, x2, y2])
            labels.append(self.categories[ann['category_id']])
            
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
            
        target = {
            'boxes': boxes,
            'labels': labels,
            'orig_size': torch.tensor([orig_h, orig_w]),
            'img_id': torch.tensor([img_id])
        }
        
        return img_tensor, target

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for handling variable number of objects."""
        images, targets = zip(*batch)
        return torch.stack(images, 0), list(targets)
