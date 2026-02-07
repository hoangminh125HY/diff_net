import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.detector import DiffNet, create_model
from utils.datasets import CocoDataset
from utils.metrics import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate DiffNet Object Detector')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='base', help='Model config name')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--data-root', type=str, default='heavyRain-1', help='Dataset root directory')
    parser.add_argument('--img-size', type=int, default=224, help='Input image size')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold')
    return parser.parse_args()


def evaluate():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    print(f"Loading validation dataset from {args.data_root}...")
    val_root = os.path.join(args.data_root, 'valid')
    val_ann = os.path.join(val_root, '_annotations.coco.json')
    
    val_dataset = CocoDataset(val_root, val_ann, img_size=(args.img_size, args.img_size))
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=CocoDataset.collate_fn
    )
    
    num_classes = val_dataset.num_classes
    
    # 2. Initialize Model
    print(f"Initializing model and loading checkpoint: {args.checkpoint}")
    model = create_model(model_config=args.config, num_classes=num_classes, img_size=(args.img_size, args.img_size))
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    
    # 3. Evaluator
    evaluator = Evaluator(val_ann)
    
    # 4. Evaluation Loop
    print("Running evaluation...")
    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images = images.to(device)
            
            outputs = model(images, conf_thres=args.conf_thres, iou_thres=args.iou_thres)
            detections = outputs['detections']
            
            img_ids = [t['img_id'].item() for t in targets]
            orig_sizes = [t['orig_size'].tolist() for t in targets]
            
            # Scale detections back to original size before adding to evaluator
            # Detection coordinates are currently [0, img_size]
            # Original coordinates are [0, orig_size]
            
            processed_detections = []
            for i, det in enumerate(detections):
                if det is not None:
                    h_orig, w_orig = orig_sizes[i]
                    h_in, w_in = args.img_size, args.img_size
                    
                    # Scale factor
                    sx = w_orig / w_in
                    sy = h_orig / h_in
                    
                    scaled_det = det.clone()
                    scaled_det[:, 0] *= sx # x1
                    scaled_det[:, 1] *= sy # y1
                    scaled_det[:, 2] *= sx # x2
                    scaled_det[:, 3] *= sy # y2
                    processed_detections.append(scaled_det)
                else:
                    processed_detections.append(None)
            
            evaluator.add_batch(processed_detections, img_ids, orig_sizes)
            
    # 5. Summarize
    print("\nAccumulating evaluation results...")
    evaluator.evaluate()
    
    print("\nIoU metric: bbox")
    evaluator.summarize()


if __name__ == "__main__":
    evaluate()
