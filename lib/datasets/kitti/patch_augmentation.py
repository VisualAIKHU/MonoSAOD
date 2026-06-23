import os
import re
import cv2
import glob
import copy
import random
import numpy as np
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from numpy.linalg import inv

# Import from kitti_utils
from .kitti_utils import Object3d, Calibration, get_calib_from_file


class KittiPatchAugmentor:
    """
    KITTI patch augmentor that adds car patches to training images.
    - Road mask logic intentionally disabled (ignored).
    - Uses precise transform: Source Cam -> LiDAR -> Target Cam -> project to image.
    """

    def __init__(self, patch_dir: str, mask_dir: str, patch_label_dir: str, calib_dir: str):
        self.patch_dir = patch_dir
        self.mask_dir = mask_dir  # Road segmentation masks directory
        self.patch_label_dir = patch_label_dir
        self.calib_dir = calib_dir

        # parameters
        self.alpha_min = 10
        self.min_opaque_pixels = 50
        self.max_patch_tries = 20
        self.min_road_overlap_ratio = 0.7

        # cache
        self.patch_files = self._load_and_filter_patch_files()

    # ------------- File / Label utils -------------

    def _load_and_filter_patch_files(self) -> List[str]:
        """Load all RGBA patch files and filter by truncation = 0 and occluded = 0."""
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        all_files = []
        for e in exts:
            all_files.extend(glob.glob(os.path.join(self.patch_dir, e)))
        all_files.sort()

        valid_files = []
        for patch_path in all_files:
            patch_basename = os.path.splitext(os.path.basename(patch_path))[0]
            patch_label_path = os.path.join(self.patch_label_dir, f"{patch_basename}.txt")
            if not os.path.exists(patch_label_path):
                continue
            try:
                with open(patch_label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            # Use Object3d from kitti_utils to parse
                            obj = Object3d(line)
                            if (obj.trucation == 0.0 and obj.occlusion == 0
                                    and 2.0 <= obj.pos[2] < 65.00
                                    and obj.level_str in ('Easy', 'Moderate', 'Hard')):
                                valid_files.append(patch_path)
                                break
            except Exception:
                continue
        return valid_files
    
    def _load_road_mask(self, scene_id: str) -> Optional[np.ndarray]:
        """Load road segmentation mask for the given scene."""
        mask_path = os.path.join(self.mask_dir, f"{scene_id}.png")
        if not os.path.exists(mask_path):
            return None
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            return mask
        except Exception:
            return None
        
    def _check_bbox_on_road(self, bbox: List[int], road_mask: np.ndarray, 
                           min_overlap_ratio: float = 0.7) -> bool:
        """
        Check if the bounding box is sufficiently on the road.
        
        Args:
            bbox: [x1, y1, x2, y2]
            road_mask: Binary road segmentation mask (non-zero = road)
            min_overlap_ratio: Minimum ratio of bbox area that must be on road
            
        Returns:
            bool: True if bbox is on road, False otherwise
        """
        x1, y1, x2, y2 = bbox
        H, W = road_mask.shape[:2]
        
        # Clip bbox to image boundaries
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W))
        y2 = max(0, min(y2, H))
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        # Extract the region of interest from road mask
        roi = road_mask[y1:y2, x1:x2]
        
        # Count road pixels (non-zero pixels)
        road_pixels = np.count_nonzero(roi)
        total_pixels = roi.size
        
        if total_pixels == 0:
            return False
        
        overlap_ratio = road_pixels / total_pixels
        return overlap_ratio >= min_overlap_ratio

    @staticmethod
    def _extract_scene_id(filename: str) -> Optional[str]:
        """Extract scene ID (######) from filename."""
        m = re.search(r"(\d{6})", os.path.basename(filename))
        return m.group(1) if m else None

    def _get_patch_label(self, patch_path: str) -> Optional[Object3d]:
        """Return the patch's single label as an Object3d instance."""
        base = os.path.splitext(os.path.basename(patch_path))[0]
        p = os.path.join(self.patch_label_dir, f"{base}.txt")
        if not os.path.exists(p):
            return None
        try:
            with open(p, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Use Object3d from kitti_utils
                        return Object3d(line)
        except Exception:
            pass
        return None

    @staticmethod
    def _alpha_composite(dst_bgr: np.ndarray, patch_rgba: np.ndarray, x: int, y: int) -> None:
        """Composite patch_rgba onto dst_bgr at (x,y)."""
        if patch_rgba.ndim == 2:
            patch_rgba = cv2.cvtColor(patch_rgba, cv2.COLOR_GRAY2BGRA)
        if patch_rgba.shape[2] == 3:
            patch_rgba = cv2.cvtColor(patch_rgba, cv2.COLOR_BGR2BGRA)
            # a = np.full(patch_rgba.shape[:2], 255, dtype=np.uint8)
            # patch_rgba = np.dstack([patch_rgba, a])

        H, W = dst_bgr.shape[:2]
        ph, pw = patch_rgba.shape[:2]

        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(W, x + pw)
        y1 = min(H, y + ph)
        if x1 <= x0 or y1 <= y0:
            return

        px0 = x0 - x
        py0 = y0 - y
        px1 = px0 + (x1 - x0)
        py1 = py0 + (y1 - y0)

        patch_win = patch_rgba[py0:py1, px0:px1]
        pbgr = patch_win[:, :, :3].astype(np.float32)
        pa = patch_win[:, :, 3:4].astype(np.float32) / 255.0

        dst_win = dst_bgr[y0:y1, x0:x1].astype(np.float32)
        out = pa * pbgr + (1.0 - pa) * dst_win
        dst_bgr[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)

    # ------------- Core augmentation -------------

    @staticmethod
    def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate IoU between two bounding boxes.
        
        Args:
            bbox1, bbox2: [x1, y1, x2, y2]
        
        Returns:
            IoU value between 0 and 1
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area

    @staticmethod
    def check_overlap_with_existing(new_bbox: List[float], 
                                    existing_bboxes: List[List[float]], 
                                    max_iou_threshold: float = 0.1) -> bool:
        """
        Check if new bbox overlaps with any existing bboxes.
        
        Args:
            new_bbox: [x1, y1, x2, y2] of the patch to be placed
            existing_bboxes: List of existing ground truth bboxes
            max_iou_threshold: Maximum allowed IoU (default 0.1 = 10% overlap)
        
        Returns:
            True if overlap is acceptable (IoU < threshold for all existing boxes)
            False if there's too much overlap
        """
        for existing_bbox in existing_bboxes:
            iou = KittiPatchAugmentor.calculate_iou(new_bbox, existing_bbox)
            if iou > max_iou_threshold:
                return False
        return True

    def _choose_random_patch(self, forbid_scene: Optional[str]) -> Optional[str]:
        """Randomly choose a patch not from the same scene."""
        if not self.patch_files:
            return None
        valids = []
        for p in self.patch_files:
            s = self._extract_scene_id(os.path.basename(p))
            if forbid_scene is None or s is None or s != forbid_scene:
                valids.append(p)
        return random.choice(valids) if valids else None

    def augment_image_with_labels(self, image: np.ndarray, scene_id: str, target_P2_or_calib,
                                existing_bboxes: Optional[List[List[float]]] = None,
                                max_iou_threshold: float = 0.1,
                                max_attempts: int = 40,
                                horizontal_search_steps: int = 10,
                                horizontal_search_range: float = 5.0):
        """
        Fixed version with correct 3D coordinate handling
        """
        H, W = image.shape[:2]
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 and image.shape[2] == 3 else image.copy()

        if existing_bboxes is None:
            existing_bboxes = []

        road_mask = self._load_road_mask(scene_id)
        if road_mask is None:
            return image.copy(), False, None

        # Prepare calibration
        if isinstance(target_P2_or_calib, Calibration):
            tgt_calib = target_P2_or_calib
        else:
            calib_path = os.path.join(self.calib_dir, f"{scene_id}.txt")
            tgt_calib = Calibration(calib_path)
        
        horizontal_offsets = np.linspace(-horizontal_search_range, horizontal_search_range, horizontal_search_steps)
        
        for attempt in range(max_attempts):
            patch_path = self._choose_random_patch(forbid_scene=scene_id)
            if patch_path is None:
                continue

            patch_rgba = cv2.imread(patch_path, cv2.IMREAD_UNCHANGED)
            if patch_rgba is None:
                continue
            if patch_rgba.ndim == 2:
                patch_rgba = cv2.cvtColor(patch_rgba, cv2.COLOR_GRAY2BGRA)
            elif patch_rgba.shape[2] == 3:
                patch_rgba = cv2.cvtColor(patch_rgba, cv2.COLOR_BGR2BGRA)
                # a = np.full(patch_rgba.shape[:2], 255, dtype=np.uint8)
                # patch_rgba = np.dstack([patch_rgba, a])
            elif patch_rgba.shape[2] == 4:
                # Critical: Convert RGBA to BGRA
                # cv2.imread loads PNGs but keeps RGBA channel order
                r, g, b, a = cv2.split(patch_rgba)
                patch_rgba = cv2.merge([b, g, r, a])  # RGBA -> BGRA

            # print(f"Patch shape: {patch_rgba.shape}, dtype: {patch_rgba.dtype}")
            # Save it to check colors
            # cv2.imwrite("/tmp/debug_patch.png", patch_rgba)

            patch_obj = self._get_patch_label(patch_path)
            if patch_obj is None:
                continue

            patch_scene = self._extract_scene_id(patch_path)
            if patch_scene is None:
                continue

            try:
                src_calib_path = os.path.join(self.calib_dir, f"{patch_scene}.txt")
                src_calib = Calibration(src_calib_path)
            except:
                continue

            # Get original alpha (observation angle) from source
            # This represents how the object looks relative to the camera
            original_alpha = patch_obj.alpha
            
            # Store original dimensions - these should NOT change
            orig_h, orig_w, orig_l = patch_obj.h, patch_obj.w, patch_obj.l

            # Transform to target camera coordinates
            center_src = patch_obj.pos.reshape(1, 3)
            center_lidar = src_calib.rect_to_lidar(center_src)
            center_tgt_base = tgt_calib.lidar_to_rect(center_lidar).reshape(-1)

            # Try different horizontal positions
            for x_offset in horizontal_offsets:
                # Apply horizontal offset
                center_tgt = center_tgt_base.copy()
                center_tgt[0] += x_offset
                
                # Calculate new viewing angle in target position
                new_viewing_angle = np.arctan2(center_tgt[0], center_tgt[2])
                
                # Keep the same appearance (alpha) but adjust rotation_y for new position
                # alpha = rotation_y - viewing_angle
                # Therefore: rotation_y = alpha + viewing_angle
                new_ry = original_alpha + new_viewing_angle
                
                # Normalize rotation_y to [-pi, pi]
                while new_ry > np.pi: 
                    new_ry -= 2 * np.pi
                while new_ry < -np.pi: 
                    new_ry += 2 * np.pi
                
                # Create a new object with updated position and rotation
                # IMPORTANT: Use original dimensions to avoid shrinking
                patch_obj_copy = copy.copy(patch_obj)  # Create new instance
                patch_obj_copy.cls_type = patch_obj.cls_type
                patch_obj_copy.trucation = patch_obj.trucation
                patch_obj_copy.occlusion = patch_obj.occlusion
                patch_obj_copy.alpha = original_alpha  # Keep original appearance
                patch_obj_copy.h = orig_h  # Use original dimensions
                patch_obj_copy.w = orig_w
                patch_obj_copy.l = orig_l
                patch_obj_copy.pos = center_tgt
                patch_obj_copy.ry = new_ry

                # Generate 3D corners with the correct dimensions and rotation
                corners_tgt = patch_obj_copy.generate_corners3d()

                # Project to 2D
                boxes, boxes_corner = tgt_calib.corners3d_to_img_boxes(corners_tgt.reshape(1, 8, 3))
                bbox_tgt = boxes[0]
                
                wd, hd = int(round(bbox_tgt[2] - bbox_tgt[0])), int(round(bbox_tgt[3] - bbox_tgt[1]))
                
                # Validity checks
                if wd <= 1 or hd <= 1 or bbox_tgt[0] >= W or bbox_tgt[1] >= H or bbox_tgt[2] <= 0 or bbox_tgt[3] <= 0:
                    continue

                # Mirror __getitem__'s deterministic projection checks (no-flip / no-crop affine)
                center_3d_world = patch_obj_copy.pos + np.array([0, -patch_obj_copy.h / 2, 0], dtype=np.float32)
                center_3d_proj, _ = tgt_calib.rect_to_img(center_3d_world.reshape(-1, 3))
                cx, cy = float(center_3d_proj[0, 0]), float(center_3d_proj[0, 1])

                # F4: projected 3D center must be inside the target image
                if not (0 <= cx < W and 0 <= cy < H):
                    continue

                # F5: projected 3D center must lie inside the placed 2D bbox
                if not (bbox_tgt[0] <= cx <= bbox_tgt[2] and bbox_tgt[1] <= cy <= bbox_tgt[3]):
                    continue

                # Road check
                if not self._check_bbox_on_road([int(round(v)) for v in bbox_tgt], road_mask, self.min_road_overlap_ratio):
                    continue

                # Overlap check
                if not self.check_overlap_with_existing(bbox_tgt, existing_bboxes, max_iou_threshold):
                    continue

                # Success! Apply augmentation
                patch_resized = cv2.resize(patch_rgba, (wd, hd), interpolation=cv2.INTER_LINEAR)
                self._alpha_composite(image_bgr, patch_resized, int(round(bbox_tgt[0])), int(round(bbox_tgt[1])))
                out_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                # print(f"[DEBUG] Augmenting scene {scene_id} with patch from scene {patch_scene}")

                # Create label with correct values
                new_label = {
                    'type': patch_obj_copy.cls_type,
                    'truncated': patch_obj_copy.trucation,
                    'occluded': patch_obj_copy.occlusion,
                    'alpha': round(float(original_alpha), 2),  # Use original alpha
                    'bbox': [float(v) for v in bbox_tgt],
                    'dimensions': [orig_h, orig_w, orig_l],  # Original dimensions
                    'location': [float(v) for v in center_tgt],
                    'rotation_y': round(float(new_ry), 2)
                }

                return out_img, True, new_label

        return image.copy(), False, None