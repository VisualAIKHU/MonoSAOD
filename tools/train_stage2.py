import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch
from copy import deepcopy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper_stage2 import Trainer
from lib.datasets.kitti.kitti_dataset_stage2 import KITTI_Dataset
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed


parser = argparse.ArgumentParser(description='Depth-aware Transformer for Monocular 3D Object Detection')
parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
args = parser.parse_args()


def main():
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    set_random_seed(cfg.get('random_seed', 444))

    model_name = cfg['model_name']
    output_path = os.path.join('./' + cfg["trainer"]['save_path'], model_name)
    os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(output_path, 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)

    # build dataloader
    train_loader, test_loader = build_dataloader(cfg['dataset'], KITTI_Dataset)

    # model initialization
    student_model, loss = build_model(cfg['model'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = list(map(int, cfg['trainer']['gpu_ids'].split(',')))
    
    if len(gpu_ids) == 1:
        student_model = student_model.to(device)
    else:
        student_model = torch.nn.DataParallel(student_model, device_ids=gpu_ids).to(device)
    
    optimizer = build_optimizer(cfg['optimizer'], student_model)

    # Load checkpoint first
    checkpoint_path = cfg['trainer'].get('pretrain_model', '')
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if student_model is not None and checkpoint['model_state'] is not None:
            student_model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        logger.info("==> Loaded checkpoint from '{}'".format(checkpoint_path))

    # Create teacher model AFTER loading weights into student
    teacher_model = deepcopy(student_model)
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()

    if args.evaluate_only:
        logger.info('###################  Evaluation Only  ##################')
        tester = Tester(cfg=cfg['tester'],
                        model=student_model,
                        dataloader=test_loader,
                        logger=logger,
                        train_cfg=cfg['trainer'],
                        model_name=model_name)
        tester.test()
        return
    
    # build lr scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)
    n_parameters = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {n_parameters}')
    trainer = Trainer(cfg=cfg['trainer'],
                      teacher=teacher_model,
                      student=student_model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      loss=loss,
                      model_name=model_name)

    tester = Tester(cfg=cfg['tester'],
                    model=trainer.student,
                    dataloader=test_loader,
                    logger=logger,
                    train_cfg=cfg['trainer'],
                    model_name=model_name)
    if cfg['dataset']['test_split'] != 'test':
        trainer.tester = tester

    logger.info('###################  Training  ##################')
    logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
    logger.info('Learning Rate: %f' % (cfg['optimizer']['lr']))

    trainer.train()

    if cfg['dataset']['test_split'] == 'test':
        return

    logger.info('###################  Testing  ##################')
    logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
    logger.info('Split: %s' % (cfg['dataset']['test_split']))

    tester.test()


if __name__ == '__main__':
    main()
