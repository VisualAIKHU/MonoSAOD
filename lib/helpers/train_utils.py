import numpy as np
import torch
import math
from lib.datasets.utils import angle2class
from torchvision.ops import nms, box_iou

def batch_apply_nms_torch(predictions_dict, iou_threshold=0.5, score_idx=-1):
    """
    Apply NMS to a dictionary of predictions for a batch, processed on the GPU.
    `score_idx` selects which column ranks the boxes (default -1 = legacy: last column).
    """
    for img_id, dets in predictions_dict.items():
        if not dets:
            continue

        # Convert to tensor and move to GPU if not already
        all_dets_tensor = torch.tensor(dets, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

        boxes = all_dets_tensor[:, 2:6]
        scores = all_dets_tensor[:, score_idx]
        classes = all_dets_tensor[:, 0]

        final_dets = []
        for cls_id in torch.unique(classes):
            cls_mask = (classes == cls_id)
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]

            if len(cls_boxes) == 0:
                continue

            # Use torchvision's NMS
            keep_indices = nms(cls_boxes, cls_scores, iou_threshold)
            
            # Fix: Get the original detection indices correctly
            cls_indices = torch.where(cls_mask)[0]
            kept_cls_indices = cls_indices[keep_indices]
            
            # Add kept detections
            for idx in kept_cls_indices:
                final_dets.append(dets[idx.item()])
        
        predictions_dict[img_id] = final_dets
    return predictions_dict

def apply_3d_nms(predictions, iou_3d_thresh=0.001):
    """
    Apply Non-Maximum Suppression based on 3D IoU.
    Assumes predictions are already sorted by score.
    """
    if not predictions:
        return []

    keep = []
    suppressed = [False] * len(predictions)

    for i in range(len(predictions)):
        if suppressed[i]:
            continue
        
        keep.append(predictions[i])
        # Only keep this box, don't suppress others based on it if score is too low

        for j in range(i + 1, len(predictions)):
            if suppressed[j]:
                continue
            
            # Only suppress if classes are the same
            if predictions[i][0] != predictions[j][0]:
                continue

            iou_3d = compute_3d_iou(predictions[i][6:13], predictions[j][6:13])
            if iou_3d > iou_3d_thresh:
                suppressed[j] = True
    return keep

def resize_box_to_feature_map(bbox, img_size, feat_size):
    """
    Resize 2D bounding box coordinates from image scale to feature map scale.
    Args:
        bbox: list or array of [xmin, ymin, xmax, ymax] in image scale
        img_size: tuple of (width, height) of the original image
        feat_size: tuple of (height, width) of the feature map
    Returns:
        resized_bbox: list of [xmin, ymin, xmax, ymax] in feature map scale
    """
    scale_x = feat_size[1] / img_size[0]
    scale_y = feat_size[0] / img_size[1]
    xmin, ymin, xmax, ymax = bbox
    xmin_feat = math.floor(xmin * scale_x)
    ymin_feat = math.floor(ymin * scale_y)
    xmax_feat = math.ceil(xmax * scale_x)
    ymax_feat = math.ceil(ymax * scale_y)

    return [xmin_feat, ymin_feat, xmax_feat, ymax_feat]

def compute_gap_features(feature_map, bbox):
    """
    Compute the average pooled features within the given bounding box on the feature map.
    """
    xmin, ymin, xmax, ymax = bbox
    # Ensure bbox is within feature map bounds
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(feature_map.shape[2], xmax)
    ymax = min(feature_map.shape[1], ymax)

    if xmin >= xmax or ymin >= ymax:
        return torch.zeros(feature_map.shape[0], device=feature_map.device)

    cropped_features = feature_map[:, ymin:ymax, xmin:xmax]
    if cropped_features.numel() == 0:
        return torch.zeros(feature_map.shape[0], device=feature_map.device)

    # Compute global average pooling
    gap_features = cropped_features.mean(dim=(1, 2))
    
    return gap_features

def compute_max_avg_cosine_similarity(GAP_feat_pred, GAP_feat_gt):
    """
    Vectorized computation of cosine similarity.
    """
    if not GAP_feat_pred or not GAP_feat_gt:
        return []

    max_similarities = []
    # Stack features for all predictions and GTs for vectorized computation
    # Each pred_feats_lvl is now a tensor of shape [num_preds, feat_dim]
    pred_feats_lvl0 = torch.stack([p[0] for p in GAP_feat_pred])
    pred_feats_lvl1 = torch.stack([p[1] for p in GAP_feat_pred])
    pred_feats_lvl2 = torch.stack([p[2] for p in GAP_feat_pred])
    
    # Each gt_feats_lvl is now a tensor of shape [num_gts, feat_dim]
    gt_feats_lvl0 = torch.stack([g[0] for g in GAP_feat_gt])
    gt_feats_lvl1 = torch.stack([g[1] for g in GAP_feat_gt])
    gt_feats_lvl2 = torch.stack([g[2] for g in GAP_feat_gt])

    # Compute cosine similarity matrices: [num_preds, num_gts]
    sims_lvl0 = torch.nn.functional.cosine_similarity(pred_feats_lvl0.unsqueeze(1), gt_feats_lvl0.unsqueeze(0), dim=-1)
    sims_lvl1 = torch.nn.functional.cosine_similarity(pred_feats_lvl1.unsqueeze(1), gt_feats_lvl1.unsqueeze(0), dim=-1)
    sims_lvl2 = torch.nn.functional.cosine_similarity(pred_feats_lvl2.unsqueeze(1), gt_feats_lvl2.unsqueeze(0), dim=-1)
    
    # Average similarities across levels
    avg_sims = (sims_lvl0 + sims_lvl1 + sims_lvl2) / 3.0


    
    # Find the max similarity for each prediction
    max_similarities, _ = torch.max(avg_sims, dim=1)
    
    return max_similarities.cpu().numpy().tolist()

def compute_2d_iou(box1, box2):
    """
    Compute 2D IoU between two boxes.
    
    Args:
        box1: [xmin, ymin, xmax, ymax]
        box2: [xmin, ymin, xmax, ymax]
    
    Returns:
        iou: float, IoU value
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_3d_iou(box1, box2):
    """
    Compute 3D IoU between two 3D boxes in camera coordinate system.
    
    Args:
        box1: [h, w, l, x, y, z, ry] - dimensions and location in camera coords
        box2: [h, w, l, x, y, z, ry] - dimensions and location in camera coords
    
    Returns:
        iou: float, 3D IoU value
    """
    # Extract dimensions and locations
    h1, w1, l1, x1, y1, z1, ry1 = box1
    h2, w2, l2, x2, y2, z2, ry2 = box2
    
    # Convert to corners in bird's eye view (x-z plane)
    corners1 = get_3d_box_corners(h1, w1, l1, x1, y1, z1, ry1)
    corners2 = get_3d_box_corners(h2, w2, l2, x2, y2, z2, ry2)
    
    # Compute height overlap
    y_max1 = y1
    y_min1 = y1 - h1
    y_max2 = y2
    y_min2 = y2 - h2
    
    height_overlap = max(0, min(y_max1, y_max2) - max(y_min1, y_min2))
    
    if height_overlap == 0:
        return 0.0
    
    # Compute bird's eye view intersection area (directly, without IoU ratio)
    from shapely.geometry import Polygon
    poly1 = Polygon(corners1[:4, [0, 2]])
    poly2 = Polygon(corners2[:4, [0, 2]])
    
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    
    inter_area = poly1.intersection(poly2).area
    
    if inter_area == 0:
        return 0.0
    
    # Compute 3D intersection and union volumes
    intersection_vol = inter_area * height_overlap
    vol1 = h1 * w1 * l1
    vol2 = h2 * w2 * l2
    union_vol = vol1 + vol2 - intersection_vol
    
    if union_vol == 0:
        return 0.0
    
    return intersection_vol / union_vol


def get_3d_box_corners(h, w, l, x, y, z, ry):
    """
    Get 3D box corners in camera coordinate system.
    
    Args:
        h, w, l: height, width, length of the box
        x, y, z: center location
        ry: rotation around Y-axis
    
    Returns:
        corners: (8, 3) array of corner points
    """
    # Create box corners in object coordinate system
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    
    # Rotate and translate to world coordinate system
    R = np.array([[np.cos(ry), 0, np.sin(ry)],
                  [0, 1, 0],
                  [-np.sin(ry), 0, np.cos(ry)]])
    
    corners = np.array([x_corners, y_corners, z_corners])
    corners = R @ corners
    corners[0, :] += x
    corners[1, :] += y
    corners[2, :] += z
    
    return corners.T


def filter_predictions_by_iou(predictions, gt_labels, iou_2d_thresh=0.5, iou_3d_thresh=0.25):
    """
    Filter out predictions that match ground truth based on IoU thresholds.

    Args:
        predictions: list of predictions, each is [cls_id, alpha, xmin, ymin, xmax, ymax, h, w, l, x, y, z, ry, score]
        gt_labels: list of ground truth labels with same format
        iou_2d_thresh: float, 2D IoU threshold for matching
        iou_3d_thresh: float, 3D IoU threshold for matching

    Returns:
        filtered_predictions: list of predictions that don't match any GT
    """
    if len(predictions) == 0 or len(gt_labels) == 0:
        return predictions

    filtered_predictions = []

    for pred in predictions:
        cls_pred = int(pred[0])
        box2d_pred = pred[2:6]  # [xmin, ymin, xmax, ymax]
        box3d_pred = pred[6:13]  # [h, w, l, x, y, z, ry]

        matched = False
        for gt in gt_labels:
            cls_gt = int(gt[0])
            
            # Only compute IoU for same class
            if cls_pred != cls_gt:
                continue
            
            box2d_gt = gt[2:6]
            box3d_gt = gt[6:13]
            
            # Compute 2D IoU
            iou_2d = compute_2d_iou(box2d_pred, box2d_gt)
            
            if iou_2d >= iou_2d_thresh:
                # Compute 3D IoU only if 2D IoU passes threshold
                iou_3d = compute_3d_iou(box3d_pred, box3d_gt)
                
                if iou_3d >= iou_3d_thresh:
                    matched = True
                    break
        
        if matched:
            continue

        filtered_predictions.append(pred)
    return filtered_predictions


def compute_2d_iou_batch(boxes1, boxes2):
    """
    Compute 2D IoU between two sets of boxes using PyTorch (batched/vectorized).
    
    Args:
        boxes1: Tensor of shape [N, 4] where each row is [xmin, ymin, xmax, ymax]
        boxes2: Tensor of shape [M, 4] where each row is [xmin, ymin, xmax, ymax]
    
    Returns:
        iou_matrix: Tensor of shape [N, M] containing IoU values
    """
    # boxes1: [N, 4], boxes2: [M, 4]
    # We need to compute IoU between every pair, resulting in [N, M] matrix
    
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    
    # Expand dimensions to enable broadcasting
    # boxes1: [N, 1, 4], boxes2: [1, M, 4]
    boxes1_expanded = boxes1.unsqueeze(1)  # [N, 1, 4]
    boxes2_expanded = boxes2.unsqueeze(0)  # [1, M, 4]
    
    # Compute intersection coordinates
    # max of mins for top-left, min of maxs for bottom-right
    x1 = torch.max(boxes1_expanded[:, :, 0], boxes2_expanded[:, :, 0])  # [N, M]
    y1 = torch.max(boxes1_expanded[:, :, 1], boxes2_expanded[:, :, 1])  # [N, M]
    x2 = torch.min(boxes1_expanded[:, :, 2], boxes2_expanded[:, :, 2])  # [N, M]
    y2 = torch.min(boxes1_expanded[:, :, 3], boxes2_expanded[:, :, 3])  # [N, M]
    
    # Compute intersection area
    intersection_width = torch.clamp(x2 - x1, min=0)
    intersection_height = torch.clamp(y2 - y1, min=0)
    intersection_area = intersection_width * intersection_height  # [N, M]
    
    # Compute areas of boxes1 and boxes2
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]
    
    # Expand to [N, M] for broadcasting
    area1_expanded = area1.unsqueeze(1)  # [N, 1]
    area2_expanded = area2.unsqueeze(0)  # [1, M]
    
    # Compute union area
    union_area = area1_expanded + area2_expanded - intersection_area  # [N, M]
    
    # Compute IoU, handle division by zero
    iou_matrix = intersection_area / (union_area + 1e-8)
    
    return iou_matrix

# def batch_filter_predictions_torch(predictions_dict, gt_labels_list, iou_2d_thresh=0.5, iou_3d_thresh=0.01):
#     """
#     Filter predictions for a batch - remove predictions that overlap with GT.
#     Remove predictions that have EITHER 2D IoU OR 3D IoU overlap with GT (not both required).
#     """
#     filtered_preds_dict = {}
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     for i, (img_id, preds) in enumerate(predictions_dict.items()):
#         gts = gt_labels_list[i]
        
#         # If no GT, keep all predictions (can't overlap with nothing)
#         if not gts:
#             filtered_preds_dict[img_id] = preds
#             continue
        
#         # If no predictions, return empty
#         if not preds:
#             filtered_preds_dict[img_id] = []
#             continue

#         # Convert to numpy for easier indexing
#         preds_np = np.array(preds)
#         gts_np = np.array(gts)
        
#         keep_mask = np.ones(len(preds), dtype=bool)
        
#         # Check each prediction
#         for pred_idx in range(len(preds)):
#             pred = preds_np[pred_idx]
#             pred_cls = int(pred[0])
#             pred_box_2d = pred[2:6]  # [xmin, ymin, xmax, ymax]
#             pred_box_3d = pred[6:13]  # [h, w, l, x, y, z, ry]
            
#             # Check against all GTs
#             for gt_idx in range(len(gts)):
#                 gt = gts_np[gt_idx]
#                 gt_cls = int(gt[0])
                
#                 # Only check same class
#                 if pred_cls != gt_cls:
#                     continue
                
#                 # Check 2D IoU
#                 gt_box_2d = gt[2:6]
#                 iou_2d = compute_2d_iou(pred_box_2d, gt_box_2d)
                
#                 if iou_2d >= iou_2d_thresh:
#                     # High 2D overlap, filter this prediction
#                     keep_mask[pred_idx] = False
#                     break  # No need to check other GTs
                
#                 # Check 3D IoU
#                 gt_box_3d = gt[6:13]
#                 iou_3d = compute_3d_iou(pred_box_3d, gt_box_3d)
                
#                 if iou_3d >= iou_3d_thresh:
#                     # High 3D overlap, filter this prediction
#                     keep_mask[pred_idx] = False
#                     break  # No need to check other GTs
        
#         # Keep only predictions that don't overlap with any GT
#         filtered_preds_dict[img_id] = [preds[idx] for idx in range(len(preds)) if keep_mask[idx]]
        
#     return filtered_preds_dict


def batch_filter_predictions_torch(predictions_dict, gt_labels_list, iou_2d_thresh=0.5, iou_3d_thresh=0.01):
    """
    Filter predictions for a batch - remove predictions that overlap with GT.
    Remove predictions that have EITHER 2D IoU OR 3D IoU overlap with GT (not both required).
    """
    filtered_preds_dict = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i, (img_id, preds) in enumerate(predictions_dict.items()):
        gts = gt_labels_list[i]
        
        # If no GT, keep all predictions (can't overlap with nothing)
        if not gts:
            filtered_preds_dict[img_id] = preds
            continue
        
        # If no predictions, return empty
        if not preds:
            filtered_preds_dict[img_id] = []
            continue

        # 리스트로 직접 처리 (np.array 사용 안 함)
        keep_mask = [True] * len(preds)
        
        # Check each prediction
        for pred_idx in range(len(preds)):
            pred = preds[pred_idx]
            pred_cls = int(pred[0])
            pred_box_2d = pred[2:6]  # [xmin, ymin, xmax, ymax]
            pred_box_3d = pred[6:13]  # [h, w, l, x, y, z, ry]
            
            # Check against all GTs
            for gt_idx in range(len(gts)):
                gt = gts[gt_idx]
                gt_cls = int(gt[0])
                
                # Only check same class
                if pred_cls != gt_cls:
                    continue
                
                # Check 2D IoU
                gt_box_2d = gt[2:6]
                iou_2d = compute_2d_iou(pred_box_2d, gt_box_2d)
                
                if iou_2d >= iou_2d_thresh:
                    # High 2D overlap, filter this prediction
                    keep_mask[pred_idx] = False
                    break  # No need to check other GTs
                
                # Check 3D IoU
                gt_box_3d = gt[6:13]
                iou_3d = compute_3d_iou(pred_box_3d, gt_box_3d)
                
                if iou_3d >= iou_3d_thresh:
                    # High 3D overlap, filter this prediction
                    keep_mask[pred_idx] = False
                    break  # No need to check other GTs
        
        # Keep only predictions that don't overlap with any GT
        filtered_preds_dict[img_id] = [preds[idx] for idx in range(len(preds)) if keep_mask[idx]]
        
    return filtered_preds_dict

def apply_confidence_based_2d_nms(predictions, iou_threshold=0.5):
    """
    Apply 2D NMS based on confidence scores.
    Assumes predictions are already sorted by score in descending order.
    
    Args:
        predictions: List of predictions [cls_id, alpha, xmin, ymin, xmax, ymax, h, w, l, x, y, z, ry, score]
                     Should be sorted by score (descending)
        iou_threshold: IoU threshold for suppression
    
    Returns:
        keep: List of predictions after NMS
    """
    if not predictions:
        return []
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Convert to tensor
    pred_tensor = torch.tensor(predictions, dtype=torch.float32, device=device)
    
    # Extract 2D boxes [xmin, ymin, xmax, ymax] and scores
    boxes = pred_tensor[:, 2:6]
    scores = pred_tensor[:, -1]
    classes = pred_tensor[:, 0]
    
    keep = []
    suppressed = torch.zeros(len(predictions), dtype=torch.bool, device=device)
    
    # Process each class separately
    for cls_id in torch.unique(classes):
        cls_mask = (classes == cls_id)
        cls_indices = torch.where(cls_mask)[0]
        
        # Already sorted by score, so just iterate in order
        for idx in cls_indices:
            if suppressed[idx]:
                continue
                
            keep.append(idx.item())
            
            # Compute IoU with remaining boxes of same class
            current_box = boxes[idx].unsqueeze(0)
            remaining_mask = cls_mask & ~suppressed
            remaining_indices = torch.where(remaining_mask)[0]
            
            if len(remaining_indices) == 0:
                continue
            
            remaining_boxes = boxes[remaining_indices]
            
            # Compute IoU
            ious = box_iou(current_box, remaining_boxes)[0]
            
            # Suppress overlapping boxes
            suppress_mask = ious > iou_threshold
            suppressed[remaining_indices[suppress_mask]] = True
    
    # Return kept predictions in original order
    return [predictions[i] for i in sorted(keep)]

def encode_pseudo_labels_as_targets(merged_labels, batch_size, device, calibs, img_size, cls_mean_size, max_objs=50):
    """
    Convert merged labels (GT + pseudo) back to the target format for loss calculation.
    """
    target_template = {
        'labels': torch.zeros(batch_size, max_objs, dtype=torch.long),
        'boxes': torch.zeros(batch_size, max_objs, 4),
        'depth': torch.zeros(batch_size, max_objs, 1),
        'size_3d': torch.zeros(batch_size, max_objs, 3),
        'heading_bin': torch.zeros(batch_size, max_objs, 1, dtype=torch.long),
        'heading_res': torch.zeros(batch_size, max_objs, 1),
        'boxes_3d': torch.zeros(batch_size, max_objs, 6),
        'mask_2d': torch.zeros(batch_size, max_objs, dtype=torch.bool)
    }

    for bz in range(batch_size):
        calib = calibs[bz]
        objects = merged_labels[bz]
        num_objs = min(len(objects), max_objs)



        for i in range(num_objs):
            obj = objects[i]
            cls_id = int(obj[0])
            
            # 2D box
            xmin, ymin, xmax, ymax = obj[2:6]
            w_2d, h_2d = xmax - xmin, ymax - ymin
            cx_2d, cy_2d = (xmin + xmax) / 2, (ymin + ymax) / 2
            
            # Normalize 2D box
            cx_norm = cx_2d / img_size[0]
            cy_norm = cy_2d / img_size[1]
            w_norm = w_2d / img_size[0]
            h_norm = h_2d / img_size[1]
            
            # 3D properties
            h3d, w3d, l3d = obj[6:9]
            x3d, y3d, z3d = obj[9:12]
            ry = obj[12]
            
            # Project 3D center to image plane
            center_3d = np.array([x3d, y3d - h3d/2, z3d])
            center_3d_img, _ = calib.rect_to_img(center_3d.reshape(1, 3))
            center_3d_img = center_3d_img[0]
            
            # Normalize 3D center and offsets
            center_3d_norm = center_3d_img / np.array(img_size)
            l_offset = (center_3d_img[0] - xmin) / img_size[0]
            r_offset = (xmax - center_3d_img[0]) / img_size[0]
            t_offset = (center_3d_img[1] - ymin) / img_size[1]
            b_offset = (ymax - center_3d_img[1]) / img_size[1]

            # Heading
            alpha = obj[1]
            heading_bin, heading_res = angle2class(alpha)

            # Fill target tensors
            target_template['mask_2d'][bz, i] = True
            target_template['labels'][bz, i] = cls_id
            target_template['boxes'][bz, i] = torch.tensor([cx_norm, cy_norm, w_norm, h_norm])
            target_template['depth'][bz, i] = z3d
            target_template['size_3d'][bz, i] = torch.tensor([h3d, w3d, l3d], dtype=torch.float32) - torch.from_numpy(cls_mean_size[cls_id])
            target_template['heading_bin'][bz, i] = torch.tensor(heading_bin, dtype=torch.long)
            target_template['heading_res'][bz, i] = torch.tensor(heading_res, dtype=torch.float32)
            target_template['boxes_3d'][bz, i] = torch.tensor([center_3d_norm[0], center_3d_norm[1], l_offset, r_offset, t_offset, b_offset])

    # Move to device
    for k, v in target_template.items():
        target_template[k] = v.to(device)
        
    return target_template

def decode_targets_to_list(targets, original_mask_2d, calibs_numpy, transformed_img_size, cls_mean_size, batch_size):
    """
    Decode targets from normalized format to KITTI format list.
    
    Args:
        targets: List of target dicts for each image in batch
        original_mask_2d: Boolean mask indicating valid objects [batch_size, max_objs]
        calibs_numpy: List of calibration objects for each image
        transformed_img_size: [width, height] of transformed image
        cls_mean_size: Mean size per class for decoding dimensions
        batch_size: Number of images in batch
    
    Returns:
        gt_as_list: List of lists, where each inner list contains GT objects in format:
                    [cls_id, alpha, xmin, ymin, xmax, ymax, h, w, l, x, y, z, ry, score]
    """
    from lib.datasets.utils import class2angle
    
    gt_as_list = []
    
    for i in range(batch_size):
        mask = original_mask_2d[i]
        img_gt_objects = []  # GT objects for current image
        
        if mask.any():  # Only process if there are valid objects
            # Extract GT data for current image
            gt_labels = targets[i]['labels'].cpu().numpy()
            gt_boxes = targets[i]['boxes'].cpu().numpy()  # [cx_norm, cy_norm, w_norm, h_norm]
            gt_depth = targets[i]['depth'].cpu().numpy()
            gt_size_3d = targets[i]['size_3d'].cpu().numpy()  # residuals from mean size
            gt_heading_bin = targets[i]['heading_bin'].cpu().numpy()
            gt_heading_res = targets[i]['heading_res'].cpu().numpy()
            gt_boxes_3d = targets[i]['boxes_3d'].cpu().numpy()  # [cx3d_norm, cy3d_norm, l, r, t, b]

            calib = calibs_numpy[i]
            
            # Decode each GT object
            for j in range(len(gt_labels)):
                cls_id = int(gt_labels[j])
                
                # Decode 2D bbox from normalized [cx, cy, w, h] to [xmin, ymin, xmax, ymax]
                cx_norm, cy_norm, w_norm, h_norm = gt_boxes[j]
                cx = cx_norm * transformed_img_size[0]
                cy = cy_norm * transformed_img_size[1]
                w = w_norm * transformed_img_size[0]
                h = h_norm * transformed_img_size[1]
                bbox_2d = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
                
                # Decode 3D dimensions (add mean size back to residuals)
                dimensions = gt_size_3d[j] + cls_mean_size[cls_id]
                
                # Decode depth (already in meters)
                depth = gt_depth[j, 0]  # gt_depth shape is (N, 1)
                
                # Decode 3D center position from boxes_3d
                cx3d_norm, cy3d_norm = gt_boxes_3d[j][:2]
                cx3d = cx3d_norm * transformed_img_size[0]
                cy3d = cy3d_norm * transformed_img_size[1]
                locations = calib.img_to_rect(cx3d, cy3d, depth).reshape(-1)
                locations[1] += dimensions[0] / 2  # Adjust y coordinate
                
                # Decode heading angle
                heading_bin_cls = int(gt_heading_bin[j, 0])  # gt_heading_bin shape is (N, 1)
                heading_res_val = gt_heading_res[j, 0]      # gt_heading_res shape is (N, 1)
                alpha = class2angle(heading_bin_cls, heading_res_val, to_label_format=True)
                ry = calib.alpha2ry(alpha, cx)
                
                # Create GT object in KITTI format
                gt_obj = [cls_id, alpha] + bbox_2d + dimensions.tolist() + locations.tolist() + [ry, 1.0]
                img_gt_objects.append(gt_obj)
        
        gt_as_list.append(img_gt_objects)  # Append list of GT objects for this image
    
    return gt_as_list