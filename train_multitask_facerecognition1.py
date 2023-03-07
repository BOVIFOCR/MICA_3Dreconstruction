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
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

# from jobs import train
from jobs import train_multitask_facerecognition1

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# BERNARDO
import numpy as np
import random
import socket
host_name = socket.gethostname()


if __name__ == '__main__':
    from configs.config import parse_args
    from configs.config_multitask_facerecognition import parse_args

    # BERNARDO
    print('Running on \'' + host_name + '\' machine...')

    if len(sys.argv) < 2:
        if host_name == 'duo':
            sys.argv.append('--cfg')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/8_mica_duo_MULTITASK_train=FRGC,LYHM,FLORENCE,FACEWAREHOUSE_eval=Stirling_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-lr=1e-5.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/8_mica_duo_MULTITASK_train=FRGC,LYHM,FLORENCE,FACEWAREHOUSE_eval=Stirling_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-lr=1e-5.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/9_mica_duo_MULTITASK_train=FRGC,LYHM,FLORENCE,FACEWAREHOUSE_eval=Stirling_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/9_mica_duo_MULTITASK_train=FRGC,LYHM,FLORENCE,FACEWAREHOUSE_eval=Stirling_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/10_mica_duo_MULTITASK-VALIDATION-WORKING_train=FRGC,LYHM,FLORENCE,FACEWAREHOUSE_eval=Stirling_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/10_mica_duo_MULTITASK-VALIDATION-WORKING_train=FRGC,LYHM,FLORENCE,FACEWAREHOUSE_eval=Stirling_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/10_mica_duo_MULTITASK-VALIDATION-WORKING_train=FRGC,LYHM,FLORENCE,FACEWAREHOUSE_eval=Stirling_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-6.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/11_mica_duo_MULTITASK-VALIDATION-WORKING_train=FRGC,LYHM,FLORENCE,FACEWAREHOUSE_eval=Stirling_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_lamb1=0.5_lamb2=1.0.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/11_mica_duo_MULTITASK-VALIDATION-WORKING_train=FRGC,LYHM,FLORENCE,FACEWAREHOUSE_eval=Stirling_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_lamb1=0.5_lamb2=1.0.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/12_mica_duo_MULTITASK-VALIDATION-WORKING_train=FRGC,LYHM,Stirling,FACEWAREHOUSE_eval=FLORENCE_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_lamb1=0.5_lamb2=1.0.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/12_mica_duo_MULTITASK-VALIDATION-WORKING_train=FRGC,LYHM,Stirling,FACEWAREHOUSE_eval=FLORENCE_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_lamb1=0.5_lamb2=1.0.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/12_mica_duo_MULTITASK-VALIDATION-WORKING_train=FRGC,LYHM,Stirling,FACEWAREHOUSE_eval=FLORENCE_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-6_lamb1=0.5_lamb2=1.0.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/13_mica_duo_MULTITASK-ARCFACE_train=FRGC_eval=FLORENCE_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-4_lamb1=1.0_lamb2=1.0.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/13_mica_duo_MULTITASK-ARCFACE_train=FRGC_eval=FLORENCE_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_lamb1=0.0_lamb2=1.0.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/13_mica_duo_MULTITASK-ARCFACE_train=FRGC_eval=FLORENCE_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_lamb1=1.0_lamb2=1.0.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/13_mica_duo_MULTITASK-ARCFACE_train=FRGC_eval=FLORENCE_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-6_lamb1=1.0_lamb2=1.0.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/13_mica_duo_MULTITASK-ARCFACE-NORM-MINMAX_train=FRGC_eval=FLORENCE_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_lamb1=0.0_lamb2=1.0.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/13_mica_duo_MULTITASK-ARCFACE-NORM-MINMAX_train=FRGC,LYHM,Stirling,FACEWAREHOUSE_eval=FLORENCE_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_lamb1=0.0_lamb2=1.0.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/14_mica_duo_MULTITASK-ARCFACE-NORM-MINMAX-ACC_train=FRGC,_eval=FLORENCE_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-6_lamb1=0.0_lamb2=1.0.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/15_mica_duo_MULTITASK-NEW-ARCFACE-NORM-MINMAX-ACC_train=FRGC,_eval=FLORENCE_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-6_lamb1=0.0_lamb2=1.0.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/15_mica_duo_MULTITASK-NEW-ARCFACE-NOXAVIER-NORM-MINMAX-ACC_train=FRGC_eval=FLORENCE_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-6_lamb1=0.0_lamb2=1.0.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/15_mica_duo_MULTITASK-NEW-ARCFACE-NOXAVIER-NORM-MINMAX-ACC_train=FLORENCE_eval=FRGC_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-6_lamb1=0.0_lamb2=1.0.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/15_mica_duo_MULTITASK-NEW-ARCFACE-NOXAVIER-NORM-MINMAX-ACC_train=FRGC,LYHM,Stirling,FACEWAREHOUSE_eval=FLORENCE_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-6_lamb1=0.0_lamb2=1.0.yml')
            sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/16_mica_duo_MULTITASK-ARCFACE-ACC_train=FRGC,FLORENCE_eval=20perc_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-6_lamb1=0.0_lamb2=1.0.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/16_mica_duo_MULTITASK-ARCFACE-ACC_train=FRGC,LYHM,FLORENCE_eval=20perc_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-6_lamb1=0.0_lamb2=1.0.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/16_mica_duo_MULTITASK-ARCFACE-ACC_train=FRGC,LYHM,Stirling,FLORENCE_eval=20perc_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-6_lamb1=1.0_lamb2=1.0.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/16_mica_duo_MULTITASK-ARCFACE-ACC_train=FRGC,LYHM,Stirling,FLORENCE_eval=20perc_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-6_lamb1=0.0_lamb2=1.0.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/16_mica_duo_MULTITASK-ARCFACE-ACC_train=FRGC,LYHM,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-6_lamb1=0.0_lamb2=1.0.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/16_mica_duo_MULTITASK-ARCFACE-ACC_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-6_lamb1=0.0_lamb2=1.0.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/16_mica_duo_MULTITASK-ARCFACE-ACC_train=Stirling_eval=20perc_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-6_lamb1=0.0_lamb2=1.0.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/16_mica_duo_MULTITASK-ARCFACE-ACC_train=FACEWAREHOUSE_eval=20perc_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-6_lamb1=0.0_lamb2=1.0.yml')
            # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/16_mica_duo_MULTITASK-ARCFACE-ACC_train=LYHM_eval=20perc_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-6_lamb1=0.0_lamb2=1.0.yml')

            sys.argv.append('--test_dataset')
            sys.argv.append('STIRLING')

            sys.argv.append('--checkpoint')
            sys.argv.append('')

    cfg, args = parse_args() 

    if cfg.cfg_file is not None:
        # exp_name = cfg.cfg_file.split('/')[-1].split('.')[0]                              # original
        exp_name = '.'.join(cfg.cfg_file.split('/')[-1].split('.')[0:-1])                   # Bernardo
        cfg.output_dir = os.path.join('./output', exp_name)                                 # original
        # cfg.output_dir = os.path.join('./output', exp_name) + cfg.output_dir_annotation   # Bernardo

    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.empty_cache()
    num_gpus = torch.cuda.device_count()

    # BERNARDO (from 'https://github.com/pytorch/pytorch/issues/45042#issuecomment-701115885' - on 29 Sep 2020)
    seed = 440
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # BERNARDO
    # print('train.py: num_gpus:', num_gpus, '   cfg:', cfg)
    print('train.py: num_gpus:', num_gpus)

    # mp.spawn(train, args=(num_gpus, cfg), nprocs=num_gpus, join=True)                           # Original
    # mp.spawn(train, args=(num_gpus, cfg), nprocs=1, join=True)                                  # BERNARDO
    mp.spawn(train_multitask_facerecognition1, args=(num_gpus, cfg), nprocs=1, join=True)         # BERNARDO
    # train(rank=num_gpus, world_size=num_gpus, cfg=cfg)

    exit(0)
