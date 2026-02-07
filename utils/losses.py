import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any


def bbox_iou(box1, box2, iou_type="ciou", eps=1e-7):
    """
    Calculate IoU, GIoU, DIoU, or CIoU between two sets of boxes.
    Boxes are in xyxy format.
    """
    # Get coordinates of boundaries
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    if iou_type == "iou":
        return iou

    # Convex hull
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b1_y1)
    
    if iou_type == "giou":
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    
    # Distance term
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((b1_x1 + b1_x2 - b2_x1 - b2_x2) ** 2 + (b1_y1 + b1_y2 - b2_y1 - b2_y2) ** 2) / 4
    
    if iou_type == "diou":
        return iou - rho2 / c2
    
    if iou_type == "ciou":
        v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - (rho2 / c2 + v * alpha)

    return iou


class TaskAlignedAssigner(nn.Module):
    """
    Task Aligned Assigner for matching ground truth to predictions.
    Used in YOLOv8/v10.
    """
    def __init__(self, topk=10, num_classes=80, alpha=1.0, beta=6.0):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        pd_scores: (B, N, num_classes)
        pd_bboxes: (B, N, 4) in xyxy
        anc_points: (N, 2)
        gt_labels: (B, M, 1)
        gt_bboxes: (B, M, 4)
        mask_gt: (B, M, 1)
        """
        bs, n_anchors, _ = pd_scores.shape
        n_max_boxes = gt_bboxes.shape[1]

        if n_max_boxes == 0:
            return None, None, None, None

        # Get alignment metrics
        # pd_scores_at_gt: (B, M, N)
        # iou: (B, M, N)
        iou = self._get_iou(pd_bboxes, gt_bboxes)
        
        # Gathering scores for corresponding categories
        # pd_scores: (B, N, C) -> permute -> (B, C, N)
        pd_scores = pd_scores.permute(0, 2, 1)
        # gt_labels: (B, M, 1) -> (B, M)
        gt_labels_ind = gt_labels.squeeze(-1).long()
        # pd_scores_at_gt: (B, M, N)
        batch_ind = torch.arange(bs, device=pd_scores.device).view(-1, 1).expand(-1, n_max_boxes)
        pd_scores_at_gt = pd_scores[batch_ind, gt_labels_ind]

        # Alignment metric: score^alpha * iou^beta
        align_metric = pd_scores_at_gt.pow(self.alpha) * iou.pow(self.beta)

        # Get topk candidates
        mask_in_gts = self._get_is_in_gts(anc_points, gt_bboxes)
        # Filter by in_gts
        align_metric *= mask_in_gts
        
        # Get topk indices
        # topk_metrics, topk_indices: (B, M, topk)
        topk_metrics, topk_indices = torch.topk(align_metric, self.topk, dim=-1, largest=True)
        
        # Create mask for topk
        mask_topk = torch.zeros_like(align_metric)
        mask_topk.scatter_(-1, topk_indices, 1.0)
        
        # Final mask
        mask_topk *= mask_gt
        
        # Handle cases where multiple GTs are assigned to same anchor
        # align_metric: (B, M, N)
        # Find which GT has max alignment for each anchor
        # max_align_val, max_align_ind: (B, N)
        max_align_val, max_align_ind = align_metric.max(dim=1)
        # mask_single_assign: (B, M, N)
        mask_single_assign = torch.zeros_like(align_metric)
        mask_single_assign.scatter_(1, max_align_ind.unsqueeze(1), 1.0)
        
        target_mask = mask_topk * mask_single_assign
        
        # Get target labels and boxes
        # target_mask: (B, M, N)
        # target_gt_idx: (B, N)
        target_gt_idx = target_mask.argmax(dim=1)
        
        # target_labels: (B, N)
        target_labels = gt_labels.squeeze(-1).gather(1, target_gt_idx)
        # target_bboxes: (B, N, 4)
        target_bboxes = gt_bboxes.gather(1, target_gt_idx.unsqueeze(-1).expand(-1, -1, 4))
        
        # mask_pos: (B, N)
        mask_pos = target_mask.sum(dim=1)
        
        # Zero out labels/boxes for non-positive samples
        target_labels = target_labels * mask_pos
        target_bboxes = target_bboxes * mask_pos.unsqueeze(-1)
        
        return target_labels, target_bboxes, mask_pos, target_gt_idx

    def _get_iou(self, pd_bboxes, gt_bboxes):
        """
        pd_bboxes: (B, N, 4)
        gt_bboxes: (B, M, 4)
        Returns: (B, M, N)
        """
        # (B, 1, N, 4) vs (B, M, 1, 4)
        pd_boxes = pd_bboxes.unsqueeze(1)
        gt_boxes = gt_bboxes.unsqueeze(2)
        
        # Intersection
        lt = torch.max(pd_boxes[..., :2], gt_boxes[..., :2])
        rb = torch.min(pd_boxes[..., 2:], gt_boxes[..., 2:])
        inter = (rb - lt).clamp(min=0).prod(dim=-1)
        
        # Areas
        pd_areas = (pd_boxes[..., 2] - pd_boxes[..., 0]) * (pd_boxes[..., 3] - pd_boxes[..., 1])
        gt_areas = (gt_boxes[..., 2] - gt_boxes[..., 0]) * (gt_boxes[..., 3] - gt_boxes[..., 1])
        union = pd_areas + gt_areas - inter + 1e-7
        
        return inter / union

    def _get_is_in_gts(self, anc_points, gt_bboxes):
        """
        anc_points: (N, 2)
        gt_bboxes: (B, M, 4)
        Returns: (B, M, N)
        """
        # anc_points: (1, 1, N, 2)
        points = anc_points.view(1, 1, -1, 2)
        # gt_bboxes: (B, M, 1, 4)
        gts = gt_bboxes.unsqueeze(2)
        
        is_in_gts = (points[..., 0] >= gts[..., 0]) & \
                    (points[..., 0] <= gts[..., 2]) & \
                    (points[..., 1] >= gts[..., 1]) & \
                    (points[..., 1] <= gts[..., 3])
        return is_in_gts.float()


class DetectionLoss(nn.Module):
    """
    Detection Loss for DiffNet (YOLOv10-style).
    Includes Classification, CIoU, and DFL losses.
    """
    def __init__(self, num_classes=80, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        # One-to-many assigner
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=num_classes)
        # One-to-one assigner for NMS-free branch
        self.o2o_assigner = TaskAlignedAssigner(topk=1, num_classes=num_classes)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, preds, targets):
        """
        preds: output from model['raw_detect']
        targets: list of target dicts
        """
        device = preds['cls_scores'].device
        
        # 1. One-to-Many Loss (O2M)
        loss_o2m = self._get_loss_branch(
            preds['cls_scores'], 
            preds['box_regs'], 
            preds['anchors'], 
            preds['strides_tensor'], 
            targets, 
            self.assigner
        )
        
        # 2. One-to-One Loss (O2O)
        loss_o2o = self._get_loss_branch(
            preds['one2one_cls'], 
            preds['one2one_reg'], 
            preds['anchors'], 
            preds['strides_tensor'], 
            targets, 
            self.o2o_assigner
        )
        
        total_loss = loss_o2m['loss'] + loss_o2o['loss']
        
        return {
            'loss': total_loss,
            'loss_cls': loss_o2m['loss_cls'] + loss_o2o['loss_cls'],
            'loss_box': loss_o2m['loss_box'] + loss_o2o['loss_box'],
            'loss_dfl': loss_o2m['loss_dfl'] + loss_o2o['loss_dfl'],
            'loss_o2m': loss_o2m['loss'],
            'loss_o2o': loss_o2o['loss']
        }

    def _get_loss_branch(self, pd_scores, pd_regs, anchors, strides, targets, assigner):
        """Calculate loss for a single branch (O2M or O2O)."""
        device = pd_scores.device
        
        # Decode boxes for matching
        pd_bboxes = self._decode_boxes(pd_regs, anchors, strides)
        
        # Prepare ground truth
        gt_labels, gt_bboxes, mask_gt = self._preprocess_targets(targets, device)
        
        # Assignment
        target_labels, target_bboxes, mask_pos, target_gt_idx = assigner(
            pd_scores.detach().sigmoid(),
            pd_bboxes.detach(),
            anchors,
            gt_labels,
            gt_bboxes,
            mask_gt
        )
        
        if mask_pos is None or mask_pos.sum() == 0:
            return {
                'loss': pd_scores.sum() * 0,
                'loss_cls': torch.tensor(0.0, device=device, requires_grad=True),
                'loss_box': torch.tensor(0.0, device=device, requires_grad=True),
                'loss_dfl': torch.tensor(0.0, device=device, requires_grad=True)
            }

        # 1. Classification Loss
        # target_scores: (B, N, C)
        target_scores = torch.zeros_like(pd_scores)
        fg_mask = mask_pos > 0
        batch_idx = torch.arange(pd_scores.shape[0], device=device).view(-1, 1).expand(-1, pd_scores.shape[1])[fg_mask]
        anchor_idx = torch.where(fg_mask)[1]
        
        target_cls = target_labels[fg_mask].long()
        # Calculate IoU for positive samples
        iou = bbox_iou(pd_bboxes[fg_mask], target_bboxes[fg_mask], iou_type="iou").squeeze(-1)
        target_scores[batch_idx, anchor_idx, target_cls] = iou
            
        loss_cls = self.bce(pd_scores, target_scores).sum() / max(mask_pos.sum(), 1)
        
        # 2. Box Regression Loss
        # CIoU Loss
        iou_val = bbox_iou(pd_bboxes[fg_mask], target_bboxes[fg_mask], iou_type="ciou")
        loss_box = (1.0 - iou_val).mean()
        
        # DFL Loss
        # target_ltrb: (N_pos, 4)
        target_ltrb = self._bbox2distance(anchors.unsqueeze(0).expand(pd_scores.shape[0], -1, -1)[fg_mask], 
                                         target_bboxes[fg_mask], strides.unsqueeze(0).unsqueeze(-1).expand(pd_scores.shape[0], -1, 2)[fg_mask])
        loss_dfl = self._df_loss(pd_regs[fg_mask], target_ltrb)
            
        total_loss = loss_cls * 1.0 + loss_box * 7.5 + loss_dfl * 1.5
        
        return {
            'loss': total_loss,
            'loss_cls': loss_cls,
            'loss_box': loss_box,
            'loss_dfl': loss_dfl
        }

    def _decode_boxes(self, reg_outputs, anchors, strides):
        """Decode ltrb to xyxy for matching."""
        # This part should match detecting head's DFL
        # (B, N, 4*reg_max) -> (B, N, 4)
        b, n, _ = reg_outputs.shape
        # Simple weighted sum for DFL
        reg = reg_outputs.view(b, n, 4, self.reg_max).softmax(-1)
        weight = torch.arange(self.reg_max, device=reg_outputs.device, dtype=torch.float)
        reg = (reg * weight).sum(-1) # (B, N, 4)
        
        lt = reg[..., :2]
        rb = reg[..., 2:]
        
        # xyxy
        x1y1 = anchors - lt * strides.unsqueeze(-1)
        x2y2 = anchors + rb * strides.unsqueeze(-1)
        return torch.cat([x1y1, x2y2], dim=-1)

    def _bbox2distance(self, anchors, bboxes, strides):
        """Convert xyxy boxes to ltrb distance from anchors."""
        x1, y1, x2, y2 = bboxes.chunk(4, -1)
        lt = (anchors - torch.cat([x1, y1], -1)) / strides
        rb = (torch.cat([x2, y2], -1) - anchors) / strides
        return torch.cat([lt, rb], -1).clamp(0, self.reg_max - 1.01)

    def _df_loss(self, pred_regs, target_ltrb):
        """Distribution Focal Loss."""
        # pred_regs: (N_pos, 4*reg_max) -> (N_pos, 4, reg_max)
        pred_regs = pred_regs.view(-1, 4, self.reg_max)
        
        target_left = target_ltrb.long()
        target_right = target_left + 1
        weight_left = target_right.float() - target_ltrb
        weight_right = 1.0 - weight_left
        
        loss_left = F.cross_entropy(pred_regs.view(-1, self.reg_max), target_left.view(-1), reduction='none').view(target_left.shape) * weight_left
        loss_right = F.cross_entropy(pred_regs.view(-1, self.reg_max), target_right.view(-1), reduction='none').view(target_left.shape) * weight_right
        
        return (loss_left + loss_right).mean()

    def _preprocess_targets(self, targets, device):
        """Convert list of targets to padded tensors."""
        batch_size = len(targets)
        max_boxes = max([len(t['boxes']) for t in targets]) if len(targets) > 0 else 0
        
        if max_boxes == 0:
            return torch.zeros((batch_size, 0, 1), device=device), \
                   torch.zeros((batch_size, 0, 4), device=device), \
                   torch.zeros((batch_size, 0, 1), device=device)
        
        gt_labels = torch.zeros((batch_size, max_boxes, 1), device=device)
        gt_bboxes = torch.zeros((batch_size, max_boxes, 4), device=device)
        mask_gt = torch.zeros((batch_size, max_boxes, 1), device=device)
        
        for i, t in enumerate(targets):
            n = len(t['boxes'])
            if n > 0:
                gt_labels[i, :n, 0] = t['labels'].to(device)
                gt_bboxes[i, :n, :] = t['boxes'].to(device)
                mask_gt[i, :n, 0] = 1.0
                
        return gt_labels, gt_bboxes, mask_gt
