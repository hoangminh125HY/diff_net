import os
import argparse
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.detector import DiffNet, create_model
from utils.datasets import CocoDataset
from utils.losses import DetectionLoss
from utils.config import load_config
from utils.metrics import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(description='Train DiffNet Object Detector')
    parser.add_argument('--config', type=str, default='base', help='Model config name (tiny, small, base, large)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--data-root', type=str, default='heavyRain-1', help='Dataset root directory')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--img-size', type=int, default=224, help='Input image size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--eval-interval', type=int, default=5, help='Interval of epochs to run evaluation')
    return parser.parse_args()

def train():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. Load Data
    print(f"Loading dataset from {args.data_root}...")
    train_root = os.path.join(args.data_root, 'train')
    train_ann = os.path.join(train_root, '_annotations.coco.json')
    val_root = os.path.join(args.data_root, 'valid')
    val_ann = os.path.join(val_root, '_annotations.coco.json')
    
    train_dataset = CocoDataset(train_root, train_ann, img_size=(args.img_size, args.img_size))
    val_dataset = CocoDataset(val_root, val_ann, img_size=(args.img_size, args.img_size))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=CocoDataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=CocoDataset.collate_fn
    )
    
    num_classes = train_dataset.num_classes
    print(f"Dataset loaded: {len(train_dataset)} training images, {len(val_dataset)} validation images")
    print(f"Number of classes: {num_classes}")
    
    # 2. Initialize Model
    print(f"Initializing model (config: {args.config})...")
    model = create_model(model_config=args.config, num_classes=num_classes, img_size=(args.img_size, args.img_size))
    model = model.to(device)
    
    # 3. Loss and Optimizer
    criterion = DetectionLoss(num_classes=num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 4. Training Loop
    print(f"Starting training on device: {device}")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = {
            'total': 0.0,
            'cls': 0.0,
            'box': 0.0,
            'dfl': 0.0
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, targets in pbar:
            images = images.to(device)
            # Targets are already list of dicts, we'll handle move to device inside loss
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss_dict = criterion(outputs['raw_detect'], targets)
            loss = loss_dict['loss']
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Logging
            count = pbar.n + 1
            epoch_losses['total'] += loss.item()
            epoch_losses['cls'] += loss_dict['loss_cls'].item()
            epoch_losses['box'] += loss_dict['loss_box'].item()
            epoch_losses['dfl'] += loss_dict['loss_dfl'].item()
            
            pbar.set_postfix({
                'loss': f"{epoch_losses['total'] / count:.4f}",
                'cls': f"{epoch_losses['cls'] / count:.4f}",
                'box': f"{epoch_losses['box'] / count:.4f}"
            })
            
        # Validation
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        avg_train_loss = epoch_losses['total'] / len(train_loader)
        print(f"Epoch {epoch+1} finished. Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            print(f"Saved best model to {save_path}")
            
        # Regular checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
            
        # mAP Evaluation
        if (epoch + 1) % args.eval_interval == 0:
            print(f"Epoch {epoch+1} evaluation...")
            eval_metrics = test(model, val_loader, val_ann, device, args.img_size)
            print("IoU metric: bbox")
            eval_metrics['evaluator'].summarize()

def validate(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            loss_dict = criterion(outputs['raw_detect'], targets)
            total_val_loss += loss_dict['loss'].item()
            
    return total_val_loss / len(val_loader) if len(val_loader) > 0 else 0

def test(model, val_loader, val_ann, device, img_size):
    model.eval()
    evaluator = Evaluator(val_ann)
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images, conf_thres=0.001)
            detections = outputs['detections']
            
            img_ids = [t['img_id'].item() for t in targets]
            orig_sizes = [t['orig_size'].tolist() for t in targets]
            
            # Scale detections
            processed_detections = []
            for i, det in enumerate(detections):
                if det is not None:
                    h_orig, w_orig = orig_sizes[i]
                    sx, sy = w_orig / img_size, h_orig / img_size
                    scaled_det = det.clone()
                    scaled_det[:, 0:4] *= torch.tensor([sx, sy, sx, sy], device=device)
                    processed_detections.append(scaled_det)
                else:
                    processed_detections.append(None)
            
            evaluator.add_batch(processed_detections, img_ids, orig_sizes)
            
    stats = evaluator.evaluate()
    return {
        'stats': stats,
        'evaluator': evaluator
    }

if __name__ == "__main__":
    train()
