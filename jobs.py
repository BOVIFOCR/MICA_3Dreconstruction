# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de


import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import yaml
from loguru import logger

from micalib.tester import Tester
from micalib.trainer import Trainer
from micalib.trainer_multitask_facerecognition1 import TrainerMultitaskFacerecognition1   # Bernardo
from utils import util

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


def setup(rank, world_size, port):
    # BERNARDO
    print('jobs.py: train(): setup(): rank=', rank, ', world_size=', world_size, ', port=', port)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    # BERNARDO
    print('torch.cuda.is_available()=', torch.cuda.is_available())
    print('torch.version.cuda=', torch.version.cuda)
    print('torch.distributed.is_available()=', torch.distributed.is_available())

    # BERNARDO
    print('jobs.py: train(): setup(): running \'dist.init_process_group()\'...')
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method="env://")   # Original
    # dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method="env://")     # Bernardo
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)  # Bernardo

    # BERNARDO
    print('jobs.py: train(): setup(): \'setup()\' function has finished!')

def deterministic(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    np.random.seed(rank)
    random.seed(rank)

    cudnn.deterministic = True
    cudnn.benchmark = False


def test(rank, world_size, cfg, args):
    port = np.random.randint(low=0, high=2000)
    setup(rank, world_size, 12310 + port)

    deterministic(rank)

    cfg.model.testing = True
    mica = util.find_model_using_name(model_dir='micalib.models', model_name=cfg.model.name)(cfg, rank)
    tester = Tester(nfc_model=mica, config=cfg, device=rank)
    tester.render_mesh = True

    if args.test_dataset.upper() == 'STIRLING':
        tester.test_stirling(args.checkpoint)
    elif args.test_dataset.upper() == 'NOW':
        tester.test_now(args.checkpoint)
    else:
        logger.error('[TESTER] Test dataset was not specified!')

    dist.destroy_process_group()


def train(rank, world_size, cfg):

    # BERNARDO
    print('jobs.py: started \'train(rank=', rank, ', world_size=', world_size, ', cfg=', cfg, ')\' function...')

    # BERNARDO
    print('jobs.py: train(): running \'port = np.random.randint()\'...')
    port = np.random.randint(low=0, high=2000)

    # BERNARDO
    print('jobs.py: train(): running \'setup(rank=', rank, ', world_size=', world_size, ', port=', 12310+port, ')\'...')
    setup(rank, world_size, 12310 + port)
    print('jobs.py: train(): function \'setup()\' has finished!')

    # BERNARDO
    print('jobs.py: train(): running \'logger.info()\'...')
    logger.info(f'[MAIN] output_dir: {cfg.output_dir}')
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.log_dir), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.vis_dir), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.val_vis_dir), exist_ok=True)

    with open(os.path.join(cfg.output_dir, cfg.train.log_dir, 'full_config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    # shutil.copy(cfg.cfg_file, os.path.join(cfg.output_dir, 'config.yaml'))

    deterministic(rank)

    # BERNARDO
    print('jobs.py: train(): running \'nfc = util.find_model_using_name()\'...')
    nfc = util.find_model_using_name(model_dir='micalib.models', model_name=cfg.model.name)(cfg, rank)
    trainer = Trainer(nfc_model=nfc, config=cfg, device=rank)

    # BERNARDO
    print('jobs.py: train(): running \'trainer.fit()\'...')
    trainer.fit()

    dist.destroy_process_group()



# Bernardo
def train_multitask_facerecognition1(rank, world_size, cfg):

    # BERNARDO
    print('jobs.py: started \'train(rank=', rank, ', world_size=', world_size, ', cfg=', cfg, ')\' function...')

    # BERNARDO
    print('jobs.py: train(): running \'port = np.random.randint()\'...')
    port = np.random.randint(low=0, high=2000)

    # BERNARDO
    print('jobs.py: train(): running \'setup(rank=', rank, ', world_size=', world_size, ', port=', 12310+port, ')\'...')
    setup(rank, world_size, 12310 + port)
    print('jobs.py: train(): function \'setup()\' has finished!')

    # BERNARDO
    print('jobs.py: train(): running \'logger.info()\'...')
    logger.info(f'[MAIN] output_dir: {cfg.output_dir}')
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.log_dir), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.vis_dir), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.val_vis_dir), exist_ok=True)

    with open(os.path.join(cfg.output_dir, cfg.train.log_dir, 'full_config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    # shutil.copy(cfg.cfg_file, os.path.join(cfg.output_dir, 'config.yaml'))

    deterministic(rank)

    # BERNARDO
    print('jobs.py: train(): running \'nfc = util.find_model_using_name()\'...')
    nfc = util.find_model_using_name(model_dir='micalib.models', model_name=cfg.model.name)(cfg, rank)
    # trainer = Trainer(nfc_model=nfc, config=cfg, device=rank)
    trainer = TrainerMultitaskFacerecognition1(nfc_model=nfc, config=cfg, device=rank)

    # BERNARDO
    print('jobs.py: train(): running \'trainer.fit()\'...')
    trainer.fit()

    dist.destroy_process_group()
