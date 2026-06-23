import gc
import math
import os
import tqdm
import copy

import torch
import numpy as np
import torch.nn as nn

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs, decode_detections
from lib.helpers.train_utils import (
    batch_apply_nms_torch, batch_filter_predictions_torch, resize_box_to_feature_map,
    compute_gap_features, compute_max_avg_cosine_similarity, encode_pseudo_labels_as_targets,
    apply_3d_nms, decode_targets_to_list
)
from lib.datasets.utils import class2angle
from utils import misc
from lib.helpers.permute_gt_bank import PrototypeBank, precompute_prototype_bank


class Trainer(object):
    def __init__(self,
                 cfg,
                 teacher,
                 student,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 loss,
                 model_name):
        self.cfg = cfg
        self.teacher = teacher
        self.student = student
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.best_result = 0
        self.best_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detr_loss = loss
        self.model_name = model_name
        self.output_dir = os.path.join('./' + cfg['save_path'], model_name)
        self.tester = None
        self.ema_decay = 0.999
        self.ema_decay_end = 0.9995
        self.max_epoch = 195
        self.gradient_clip = cfg.get('gradient_clip', 1.0)


        self.pseudo_bank = {}
        self.pseudo_bank_path = os.path.join(self.output_dir, "pseudo_bank_latest.pt")
        if os.path.exists(self.pseudo_bank_path):
            try:
                self.pseudo_bank = torch.load(self.pseudo_bank_path, map_location='cpu')
                self.logger.info(f"Loaded pseudo_bank from {self.pseudo_bank_path} ({len(self.pseudo_bank)} images)")
            except Exception as e:
                self.logger.warning(f"Failed to load pseudo_bank: {e}")


        # Load pretrained model
        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_checkpoint(model=self.student,
                            optimizer=None,
                            filename=cfg['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger)

        # Resume training
        if cfg.get('resume_model', None):
            resume_model_path = os.path.join(self.output_dir, "checkpoint.pth")
            assert os.path.exists(resume_model_path)
            self.epoch, self.best_result, self.best_epoch = load_checkpoint(
                model=self.student.to(self.device),
                optimizer=self.optimizer,
                filename=resume_model_path,
                map_location=self.device,
                logger=self.logger)
            self.lr_scheduler.last_epoch = self.epoch - 1
            self.logger.info(f"Loading Checkpoint... Best Result:{self.best_result}, Best Epoch:{self.best_epoch}")

        # Precompute prototype bank (GT 기반 초기화)
        self.prototype_bank = precompute_prototype_bank(self.train_loader, self.teacher, device=self.device)
        self.sim_threshold = 0.8
        self.ema_alpha_online = 0.005


    def update_pseudo_bank(self, pseudo_labels, info):
        """Update memory-based pseudo label bank."""
        for i, img_id in enumerate(info['img_id']):
            img_id = int(img_id.item()) if torch.is_tensor(img_id) else int(img_id)
            new_labels = pseudo_labels[i]
            if len(new_labels) == 0:
                continue
            if img_id not in self.pseudo_bank:
                self.pseudo_bank[img_id] = []
            self.pseudo_bank[img_id].extend(new_labels)

    def merge_with_pseudo_bank(self, gt_as_list, info):
        """Merge current GTs with previous pseudo GTs stored in memory."""
        merged_labels = []
        for i, img_id in enumerate(info['img_id']):
            img_id = int(img_id.item()) if torch.is_tensor(img_id) else int(img_id)
            gt_list = gt_as_list[i]
            pseudo_prev = self.pseudo_bank.get(img_id, [])
            merged_labels.append(gt_list + pseudo_prev)
        return merged_labels


    def get_ema_decay(self, epoch):
        progress = min(epoch / self.max_epoch, 1.0)
        ema_decay = self.ema_decay + (self.ema_decay_end - self.ema_decay) * progress
        return ema_decay

    def update_teacher_ema(self):
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
                teacher_param.data.mul_(self.ema_decay).add_(student_param.data, alpha=1 - self.ema_decay)

    def train(self):
        start_epoch = self.epoch
        progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']), dynamic_ncols=True, desc='epochs')
        best_result = self.best_result
        best_epoch = self.best_epoch

        for epoch in range(start_epoch, self.cfg['max_epoch']):
            if hasattr(self.train_loader.dataset, 'set_epoch'):
                self.train_loader.dataset.set_epoch(epoch)
            np.random.seed(np.random.get_state()[1][0] + epoch)
            self.train_one_epoch(epoch)
            self.epoch += 1

            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            if (self.epoch % self.cfg['save_frequency']) == 0:
                os.makedirs(self.output_dir, exist_ok=True)
                ckpt_name = os.path.join(self.output_dir, 'checkpoint')
                save_checkpoint(
                    get_checkpoint_state(self.student, self.optimizer, self.epoch, best_result, best_epoch),
                    ckpt_name)

                if self.tester is not None:
                    self.logger.info(f"Test Epoch {self.epoch}")
                    self.tester.inference()
                    cur_result = self.tester.evaluate()
                    if cur_result > best_result:
                        best_result = cur_result
                        best_epoch = self.epoch
                        ckpt_name = os.path.join(self.output_dir, 'checkpoint_best')
                        save_checkpoint(
                            get_checkpoint_state(self.student, self.optimizer, self.epoch, best_result, best_epoch),
                            ckpt_name)
                    self.logger.info(f"Best Result:{best_result}, epoch:{best_epoch}")

            progress_bar.update()

        self.logger.info(f"Best Result:{best_result}, epoch:{best_epoch}")

    def train_one_epoch(self, epoch):
        torch.set_grad_enabled(True)
        self.student.train()
        self.teacher.eval()
        total_added_epoch = 0

        self.ema_decay = self.get_ema_decay(epoch)
        print(f">>>>>>> Epoch: {epoch}")
        
        progress_bar = tqdm.tqdm(total=len(self.train_loader), desc='iters', leave=False)
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.train_loader):
            inputs_student, inputs_teacher = inputs
            inputs_student = inputs_student.to(self.device)
            inputs_teacher = inputs_teacher.to(self.device)
            calibs = calibs.to(self.device)

            for key in targets.keys():
                targets[key] = targets[key].to(self.device)
            img_sizes = targets['img_size']
            transformed_img_size = [inputs_teacher.shape[3], inputs_teacher.shape[2]]
            original_mask_2d = targets['mask_2d']
            full_gt_kitti = targets['full_gt_kitti'].detach().cpu().numpy()
            full_gt_mask = targets['full_gt_mask'].detach().cpu().numpy()
            targets = self.prepare_targets(targets, inputs_teacher.shape[0])

            dn_args = None
            if self.cfg.get("use_dn"):
                dn_args = (
                    targets,
                    self.cfg['scalar'],
                    self.cfg['label_noise_scale'],
                    self.cfg['box_noise_scale'],
                    self.cfg['num_patterns']
                )

            self.optimizer.zero_grad()

            # --- Teacher forward ---
            with torch.no_grad():
                # >>> CHANGED: teacher가 (predictions, features, srcs) 를 반환해야 함
                predictions, features, srcs = self.teacher(
                    inputs_teacher, calibs, None, img_sizes, dn_args=dn_args
                )

                dets_pred = extract_dets_from_outputs(outputs=predictions, K=50, topk=50)
                dets_pred = dets_pred.detach().cpu().numpy()
                calibs_numpy = [self.train_loader.dataset.get_calib(idx) for idx in info['img_id']]
                info_numpy = {k: v.detach().cpu().numpy() for k, v in info.items()}
                info_for_decode = info_numpy.copy()
                info_for_decode['img_size'] = np.array([
                    [inputs_teacher.shape[3], inputs_teacher.shape[2]] for _ in range(inputs_teacher.shape[0])
                ])

                cls_mean_size = self.train_loader.dataset.cls_mean_size
                predictions = decode_detections(
                    dets=dets_pred,
                    info=info_for_decode,
                    calibs=calibs_numpy,
                    cls_mean_size=cls_mean_size,
                    threshold=0.2,
                    return_uncertainty=True,
                )

                # NMS ranks by the classification confidence column (score), not by depth confidence.
                # decode_detections(return_uncertainty=True) layout: [..., ry, score, 1/σ²], so score is at index -2.
                predictions_dict_2d_nms = batch_apply_nms_torch(predictions, iou_threshold=0.7, score_idx=-2)
                predictions_dict_nms = {}
                for img_id, dets in predictions_dict_2d_nms.items():
                    sorted_dets = sorted(dets, key=lambda x: x[-2], reverse=True)
                    predictions_dict_nms[img_id] = apply_3d_nms(sorted_dets, iou_3d_thresh=0.25)

                gt_as_list = decode_targets_to_list(
                    targets, original_mask_2d, calibs_numpy, transformed_img_size,
                    cls_mean_size, len(info['img_id'])
                )

                gt_as_list = self.merge_with_pseudo_bank(gt_as_list, info)
                pseudo_candidates = batch_filter_predictions_torch(predictions_dict_nms, gt_as_list)

                # --- Use prototype bank for pseudo selection ---
                pseudo_labels = []
                bank = self.prototype_bank
                GAP_feat_gt, GAP_feat_pred = None, None

                for i in range(len(info['img_id'])):
                    img_id = info['img_id'][i].item()
                    pred_dets = pseudo_candidates.get(img_id, [])
                    current_gt_objects = gt_as_list[i]

                    if len(pred_dets) == 0 :
                        pseudo_labels.append([])
                        continue

                    sigma = [det[-1] for det in pred_dets]

                    per_img_feats = [srcs[0][i], srcs[1][i], srcs[2][i]]

                    # --- Update bank with GT features (mean-fused across 4 levels)
                    GAP_feat_gt = [
                        tuple(
                            compute_gap_features(
                                f,
                                resize_box_to_feature_map(obj[2:6], transformed_img_size, (f.shape[1], f.shape[2]))
                            )
                            for f in per_img_feats
                        )
                        for obj in current_gt_objects
                    ]

                    bank.update_gt_features(GAP_feat_gt, alpha=self.ema_alpha_online)

                    # --- Compute pseudo scores (mean-fused across 4 levels)
                    GAP_feat_pred = [
                        tuple(
                            compute_gap_features(
                                f,
                                resize_box_to_feature_map(det[2:6], transformed_img_size, (f.shape[1], f.shape[2]))
                            )
                            for f in per_img_feats
                        )
                        for det in pred_dets
                    ]

                    pl_scores = bank.get_scores(GAP_feat_pred)

                    # if epoch % 5 == 0 :
                    #     VIS_SCENE_ID = [99, 172, 254, 259, 442, 597, ]
                    #     should_visualize = any(img_id.item() in VIS_SCENE_ID for img_id in info['img_id'])
                    #     if should_visualize:
                    #         with open(f"./debug/pseudo_labels_epoch.txt", "a") as f:
                    #                 f.write(f"{batch_idx}, id : {img_id}, ps : {pl_scores} .\n")

                    positive = [pred_dets[j] for j, sim in enumerate(pl_scores) if sim > 0.85 and sigma[j] > 1]
                    positive_indices = [j for j, s in enumerate(pl_scores) if s > 0.85 and sigma[j] > 1]
                    pseudo_labels.append(positive)

                    if len(positive_indices) > 0:
                        pos_feats = [GAP_feat_pred[j] for j in positive_indices]
                        bank.update_gt_features(pos_feats, alpha=self.ema_alpha_online)


                merged_labels = [(gt_as_list[k] + pseudo_labels[k]) for k in range(len(gt_as_list))]

                self.update_pseudo_bank(pseudo_labels, info)

                num_added = sum(len(p) for p in pseudo_labels if len(p) > 0)
                total_added_epoch += num_added

                num_added = sum(len(p) for p in pseudo_labels if len(p) > 0)
                num_total = sum(len(v) for v in self.pseudo_bank.values())

                new_targets = encode_pseudo_labels_as_targets(
                    merged_labels,
                    inputs_teacher.shape[0],
                    self.device,
                    calibs_numpy,
                    transformed_img_size,
                    cls_mean_size,
                    max_objs=self.train_loader.dataset.max_objs
                )
                new_targets = self.prepare_targets(new_targets, inputs_teacher.shape[0])

            # >>> CHANGED: 정리할 변수에 srcs 포함
            predictions = features = srcs= None
            del GAP_feat_pred, GAP_feat_gt
            del info_for_decode, gt_as_list, targets

            # --- Student training ---
            outputs, _, _ = self.student(inputs_student, calibs, new_targets, img_sizes, dn_args=dn_args)
            detr_losses_dict = self.detr_loss(outputs, new_targets, mask_dict=None)

            weight_dict = self.detr_loss.weight_dict
            detr_losses = sum(detr_losses_dict[k] * weight_dict[k] for k in detr_losses_dict if k in weight_dict)

            detr_losses.backward()
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.gradient_clip)
            self.optimizer.step()
            self.update_teacher_ema()

            progress_bar.update()
            torch.cuda.empty_cache()
            gc.collect()

        progress_bar.close()


        num_total = sum(len(v) for v in self.pseudo_bank.values())
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "pseudo_bank_stats.txt"), "a") as f:
            f.write(f"Epoch {epoch:03d}  |  Added: {total_added_epoch:6d}  |  Total: {num_total:6d}\n")

    def prepare_targets(self, targets, batch_size):
        targets_list = []
        mask = targets['mask_2d']
        key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin', 'heading_res', 'boxes_3d']

        for bz in range(batch_size):
            target_dict = {}
            for key, val in targets.items():
                if key in key_list:
                    target_dict[key] = val[bz][mask[bz]]
            targets_list.append(target_dict)
        return targets_list
