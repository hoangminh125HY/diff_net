import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os


class Evaluator:
    """
    Evaluator for object detection using COCO metrics.
    """
    def __init__(self, ann_file):
        """
        Args:
            ann_file: Path to COCO annotation JSON file.
        """
        self.coco_gt = COCO(ann_file)
        self.results = []
        
    def add_batch(self, detections, img_ids, orig_sizes):
        """
        Add batch results for evaluation.
        
        Args:
            detections: List of tensors [x1, y1, x2, y2, conf, cls] (one per image)
            img_ids: List of image IDs (tensors or ints)
            orig_sizes: List of original image sizes [h, w]
        """
        for i, det in enumerate(detections):
            if det is None or len(det) == 0:
                continue
                
            img_id = int(img_ids[i])
            h_orig, w_orig = orig_sizes[i]
            
            # The model predicts on img_size (e.g. 224x224)
            # We need to scale back to original size if detections are in resized space
            # In our implementation, CocoDataset handles scaling in __getitem__
            # So detections here are in the space of input image (e.g. 224x224)
            # We need to scale back to original image size for COCO eval
            
            # Assuming input size was H_in, W_in (e.g. 224, 224)
            # We need to know this H_in, W_in to scale back.
            # For now, let's assume detection coords are [0, img_size]
            
            # TODO: Pass input_size to evaluator or scale before calling add_batch
            
            det = det.cpu().numpy()
            for x1, y1, x2, y2, conf, cls_id in det:
                # coco bbox: [x, y, w, h]
                w = x2 - x1
                h = y2 - y1
                
                # Fetch category ID from index (mapping index -> coco cat_id)
                # In CocoDataset, cat_id were mapped to 0..N-1
                # We need the inverse mapping or use the original cat_ids
                # For simplicity, if training/eval use same loader, indexing is consistent
                # but COCO eval needs the REAL cat_id from JSON
                
                cat_ids = self.coco_gt.getCatIds()
                # cat_ids is sorted by default in our loader implementation?
                # cats = {cat['id']: i for i, cat in enumerate(self.coco_data['categories'])}
                # So cat_ids[cls_id] matches!
                
                res = {
                    "image_id": img_id,
                    "category_id": cat_ids[int(cls_id)],
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(conf),
                }
                self.results.append(res)
                
    def evaluate(self):
        """
        Perform evaluation and return metrics.
        """
        if len(self.results) == 0:
            print("No detections found for evaluation.")
            return {}
            
        # Save results to temporary file
        res_file = "temp_results.json"
        with open(res_file, "w") as f:
            json.dump(self.results, f)
            
        self.coco_dt = self.coco_gt.loadRes(res_file)
        self.coco_eval = COCOeval(self.coco_gt, self.coco_dt, 'bbox')
        self.coco_eval.evaluate()
        self.coco_eval.accumulate()
        
        # Populate stats by calling summarize with output suppressed
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            self.coco_eval.summarize()
            
        # Clean up
        if os.path.exists(res_file):
            os.remove(res_file)
            
        return {
            "AP": self.coco_eval.stats[0],
            "AP50": self.coco_eval.stats[1],
            "AP75": self.coco_eval.stats[2],
            "AP_S": self.coco_eval.stats[3],
            "AP_M": self.coco_eval.stats[4],
            "AP_L": self.coco_eval.stats[5],
            "AR@1": self.coco_eval.stats[6],
            "AR@10": self.coco_eval.stats[7],
            "AR@100": self.coco_eval.stats[8],
            "AR_S": self.coco_eval.stats[9],
            "AR_M": self.coco_eval.stats[10],
            "AR_L": self.coco_eval.stats[11],
        }

    def summarize(self):
        """
        Print COCO summary.
        """
        if hasattr(self, 'coco_eval'):
            self.coco_eval.summarize()
