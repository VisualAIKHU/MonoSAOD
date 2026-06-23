import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from lib.helpers.prototype import TwoStageBin
from lib.helpers.train_utils import (
    resize_box_to_feature_map, compute_gap_features,
    decode_targets_to_list
)

# ============================================================
# PrototypeBank: 단일 통합형 프로토타입 뱅크
# ============================================================
class PrototypeBank:
    def __init__(self, dim=256, K=192, m=1024, alpha=0.05,
                 dtype=torch.float16, device='cuda'):
        """
        PrototypeBank (Unified)
        - 하나의 TwoStageBin으로 모든 feature를 통합 저장.
        - rotation, depth 등 구분 없음.
        - 기존 rotation-bin 구조의 4배 크기.
        """
        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.alpha = alpha
        self.sim_threshold = 0.9
        self.bank = TwoStageBin(dim=dim, K=K, m=m,
                                alpha=alpha, dtype=dtype, device=device)

    # -------------------------------------------------------------
    # GT feature 업데이트
    # -------------------------------------------------------------
    def update_gt_features(self, gap_features, alpha=0.005):
        """
        gap_features: list of [ [lvl0_feat, lvl1_feat], ... ] per object
        """
        τ_new = 0.8
        for multi_level_feats in gap_features:
            valid_feats = [f for f in multi_level_feats if f is not None]
            if len(valid_feats) == 0:
                continue

            fused_feat = torch.stack(valid_feats, dim=0).mean(0)
            fused_feat = fused_feat / (fused_feat.norm() + 1e-12)
            bin_obj = self.bank
            bin_obj.alpha = alpha

            if bin_obj.M.size(0) == 0:
                bin_obj.M = fused_feat[None].clone()
                bin_obj.count = torch.tensor([1], dtype=torch.int32, device=self.device)
                bin_obj.members = [fused_feat[None].to(self.dtype)]
                continue

            sims = torch.mm(bin_obj.M, fused_feat.unsqueeze(1)).squeeze(1)
            high_sim = torch.where(sims >= τ_new)[0]

            if len(high_sim) == 0 and bin_obj.M.size(0) < bin_obj.K:
                bin_obj.M = torch.cat([bin_obj.M, fused_feat[None]], dim=0)
                bin_obj.count = torch.cat(
                    [bin_obj.count, torch.tensor([1], dtype=torch.int32, device=self.device)]
                )
                bin_obj.members.append(fused_feat[None].to(bin_obj.dtype))
            elif len(high_sim) > 0:
                for idx in high_sim:
                    mu = bin_obj.M[idx]
                    mu = (1 - bin_obj.alpha) * mu + bin_obj.alpha * fused_feat
                    mu = mu / (mu.norm() + bin_obj.eps)
                    bin_obj.M[idx] = mu
                    bin_obj.count[idx] += 1
                    bin_obj._add_member(idx, fused_feat)
            else:
                best_idx = int(torch.argmax(sims))
                mu = bin_obj.M[best_idx]
                mu = (1 - bin_obj.alpha) * mu + bin_obj.alpha * fused_feat
                mu = mu / (mu.norm() + bin_obj.eps)
                bin_obj.M[best_idx] = mu
                bin_obj.count[best_idx] += 1
                bin_obj._add_member(best_idx, fused_feat)

    # -------------------------------------------------------------
    # query feature 점수 계산
    # -------------------------------------------------------------
    def get_scores(self, query_features):
        """
        query_features: list of [ [lvl0_feat, lvl1_feat], ... ]
        """
        scores = []
        for multi_level_feats in query_features:
            valid_feats = [f for f in multi_level_feats if f is not None]
            if len(valid_feats) == 0:
                scores.append(0.0)
                continue

            fused_feat = torch.stack(valid_feats, dim=0).mean(0)
            fused_feat = fused_feat / (fused_feat.norm() + 1e-12)
            bin_obj = self.bank
            if bin_obj.M.size(0) == 0:
                scores.append(0.0)
                continue

            sims = torch.mm(bin_obj.M, fused_feat.unsqueeze(1)).squeeze(1)
            scores.append(sims.max().item())

        return torch.tensor(scores, device=self.device)


# ============================================================
# Precompute Stage
# ============================================================
def precompute_prototype_bank(train_loader, teacher, device='cuda'):
    print("\n[Stage 1] Building Unified Prototype Bank...\n")
    teacher.eval()
    bank = PrototypeBank(dim=256, K=256, m=512, alpha=0.05, device=device)

    for batch_idx, (inputs, calibs, targets, info) in enumerate(tqdm(train_loader, desc="Precomputing Bank")):
        inputs_teacher = inputs[1].to(device)
        calibs = calibs.to(device)
        for k in targets.keys():
            targets[k] = targets[k].to(device)

        img_sizes = targets['img_size']
        transformed_img_size = [inputs_teacher.shape[3], inputs_teacher.shape[2]]
        original_mask_2d = targets['mask_2d']
        calibs_numpy = [train_loader.dataset.get_calib(i) for i in info['img_id']]
        cls_mean_size = train_loader.dataset.cls_mean_size
        prepared_targets = prepare_targets(targets, inputs_teacher.shape[0])

        with torch.no_grad():
            outputs, features, srcs = teacher(inputs_teacher, calibs, None, img_sizes)
            gt_as_list = decode_targets_to_list(
                prepared_targets, original_mask_2d, calibs_numpy, transformed_img_size,
                cls_mean_size, len(info['img_id'])
            )

        for i, gt_objects in enumerate(gt_as_list):
            if len(gt_objects) == 0:
                continue

            feat_lv0, feat_lv1, feat_lv2 = srcs[0][i], srcs[1][i],srcs[2][i]
            GAP_feat_gt = [
                tuple(
                    compute_gap_features(f, resize_box_to_feature_map(obj[2:6],
                                            transformed_img_size, (f.shape[1], f.shape[2])))
                    for f in [feat_lv0, feat_lv1, feat_lv2]
                )
                for obj in gt_objects
            ]
            bank.update_gt_features(GAP_feat_gt, alpha=0.01)

    print("\n[Stage 1 Complete] Unified Prototype Bank built successfully.\n")
    print_bank_structure(bank)
    return bank


# ============================================================
# Helper
# ============================================================
def prepare_targets(targets, batch_size):
    targets_list = []
    mask = targets['mask_2d']
    key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d',
                'heading_bin', 'heading_res', 'boxes_3d']
    for bz in range(batch_size):
        target_dict = {}
        for key, val in targets.items():
            if key in key_list:
                target_dict[key] = val[bz][mask[bz]]
        targets_list.append(target_dict)
    return targets_list


def print_bank_structure(bank):
    print("==== PrototypeBank Structure ====")
    num_proto = bank.bank.M.size(0)
    num_members = sum([m.size(0) for m in bank.bank.members]) if num_proto > 0 else 0
    print(f"Total prototypes: {num_proto}")
    print(f"Total members: {num_members}")
    print("=================================")
