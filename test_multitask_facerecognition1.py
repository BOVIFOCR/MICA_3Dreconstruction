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

# from jobs import test                            # original
from jobs import test_multitask_facerecognition1   # Bernardo

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


configs_folder = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs'
models_folder = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/output'


if __name__ == '__main__':
    # from configs.config import parse_args
    from configs.config_multitask_facerecognition import parse_args

    # # Original ARCFACE (no MICA train, sanity check)
    # model = '19_mica_duo_pretrainedARCFACE=ms1mv3-r100_fr-feat=original-arcface_ORIGINAL-ARCFACE'
    # checkpoint = ''    # LFW: 99.8%,  MLFW: 91.8%,  TALFW: 68.2%

    # ARCFACE (2D only)
    # model = '16_mica_duo_MULTITASK-ARCFACE-ACC-CONFMAT_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=arcface_fr-lr=1e-7_wd=2e-5_lamb1=0.0_lamb2=1.0'
    # checkpoint = 'model_300000.tar'
    # model = '16_mica_duo_MULTITASK-ARCFACE-ACC-CONFMAT_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=arcface_fr-lr=1e-7_wd=2e-5_lamb1=0.0_lamb2=1.0'
    # checkpoint = 'model_350000.tar'
    # model = '20_SINGLE-TASK-ARCFACE-ACC-CONFMAT_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=arcface_fr-lr=1e-5_wd=1e-5_opt=SGD_sched=CosAnn_reset-opt=True_lamb1=0.0_lamb2=1.0'
    # checkpoint = 'model_190000.tar'      # LFW: 95.3%,  MLFW: 68.5%,  TALFW: 75.2%
    # model = '20_SINGLE-TASK-ARCFACE-ACC-CONFMAT_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=arcface_fr-lr=1e-5_wd=1e-5_opt=SGD_sched=CosAnn_reset-opt=True_lamb1=0.0_lamb2=1.0'
    # checkpoint = 'model_10000.tar'       # LFW: 98.5%,  MLFW: 81.9%,  TALFW: 70.0%

    # # Multi-task (ArcFace + Reconstruction)
    # model = '16_mica_duo_MULTITASK-ARCFACE-ACC-CONFMAT_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=arcface_lr=1e-5_arc-lr=1e-5_fr-lr=1e-7_wd=2e-5_lamb1=1.0_lamb2=1.0'
    # checkpoint = 'model_20000.tar'
    # model = '16_mica_duo_MULTITASK-ARCFACE-ACC-CONFMAT_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=arcface_lr=1e-5_arc-lr=1e-5_fr-lr=1e-8_wd=2e-5_lamb1=0.1_lamb2=1.0'
    # checkpoint = 'model_20000.tar'
    # model = '24_PLOT-GRAD-ANGLES_SUM-LOSSES_mica_duo_MULTITASK_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=arcface_lr=1e-5_arc-lr=1e-5_fr-lr=1e-5_wd=2e-5_lamb1=0.02_lamb2=0.98'
    # checkpoint = 'model_30000.tar'     # LFW: 98.8%,  MLFW: 81.9%,  TALFW: 71.2%
    # model = '24_PLOT-GRAD-ANGLES_SUM-LOSSES_mica_duo_MULTITASK_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=arcface_lr=1e-5_arc-lr=1e-5_fr-lr=1e-5_wd=2e-5_lamb1=0.01_lamb2=0.99'
    # checkpoint = 'model_40000.tar'     # LFW: 99.0%,  MLFW: 82.4%,  TALFW: 71.0%
    # model = '26_SANITY-CHECK_MULTI-TASK_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=arcface_fr-lr=1e-5_loss=arcface_marg1=0.5_scal1=32_wd=1e-5_opt=SGD_sched=CosAnn_reset-opt=True_lamb1=1.0_lamb2=1.0'
    # checkpoint = 'model_10000.tar'     # LFW: 99.4%,  MLFW: 83.3%,  TALFW: 73.1%
    # checkpoint = 'model_20000.tar'     # LFW: 99.4%,  MLFW: 82.2%,  TALFW: 74.1%
    # checkpoint = 'model_30000.tar'     # LFW: 99.5%,  MLFW: 81.5%,  TALFW: 74.1%
    # checkpoint = 'model_120000.tar'    # LFW: 98.8%,  MLFW: 79.0%,  TALFW: 74.5%
    # checkpoint = 'model_200000.tar'    # LFW: 98.8%,  MLFW: 77.6%,  TALFW: 75.2%

    # # 3DMM (3D only)
    # model = '16_mica_duo_MULTITASK-ARCFACE-ACC-CONFMAT_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-7_wd=2e-5_lamb1=0.0_lamb2=1.0'
    # checkpoint = 'model_300000.tar'
    # model = '16_mica_duo_MULTITASK-ARCFACE-ACC-CONFMAT_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-7_wd=2e-5_lamb1=0.0_lamb2=1.0'
    # checkpoint = 'model_250000.tar'
    # model = '20_SINGLE-TASK-ARCFACE-ACC-CONFMAT_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_wd=1e-5_opt=SGD_sched=CosAnn_reset-opt=True_lamb1=0.0_lamb2=1.0'
    # checkpoint = 'model_180000.tar'   # LFW: 92.2%,  MLFW: 63.9%,  TALFW: 73.5%
    # checkpoint = 'model_210000.tar'   # LFW: 91.7%,  MLFW: 62.7%,  TALFW: 74.4%
    # model = '20_SINGLE-TASK-ARCFACE-ACC-CONFMAT_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_wd=1e-5_opt=SGD_sched=CosAnn_reset-opt=True_lamb1=0.0_lamb2=1.0'
    # checkpoint = 'model_10000.tar'    # LFW: 92.5%,  MLFW: 72.4%,  TALFW: 64.3%
    # checkpoint = 'model_20000.tar'    # LFW: 91.9%,  MLFW: 72.3%,  TALFW: 63.9%

    # Multi-task (3DMM + Reconstruction)
    # model = '16_mica_duo_MULTITASK-ARCFACE-ACC-CONFMAT_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_lr=1e-5_arc-lr=1e-5_fr-lr=1e-7_wd=2e-5_lamb1=1.0_lamb2=1.0'
    # checkpoint = 'model_200000.tar'
    # model = '16_mica_duo_MULTITASK-ARCFACE-ACC-CONFMAT_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_lr=1e-5_arc-lr=1e-5_fr-lr=1e-7_wd=2e-5_lamb1=0.1_lamb2=1.0'
    # checkpoint = 'model_250000.tar'
    # model = '20_MULTITASK-ARCFACE-ACC-CONFMAT_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_lr=1e-5_arc-lr=1e-5_fr-lr=1e-7_wd=1e-5_opt=AdamW_reset-opt=True_lamb1=1.0_lamb2=1.0'
    # checkpoint = 'model_290000.tar'
    # model = '20_MULTITASK-ARCFACE_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_lr=1e-5_arc-lr=1e-5_fr-lr=1e-5_wd=1e-6_opt=SGD_sched=CosAnn_reset-opt=True_lamb1=1.0_lamb2=1.0'
    # checkpoint = 'model_80000.tar'
    # model = '20_MULTITASK-ARCFACE_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_lr=1e-5_arc-lr=1e-5_fr-lr=5e-5_wd=1e-6_opt=SGD_sched=CosAnn_reset-opt=True_lamb1=1.0_lamb2=1.0'
    # checkpoint = 'model_150000.tar'
    # model = '20_MULTITASK-ARCFACE_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_lr=1e-5_arc-lr=1e-5_fr-lr=1e-5_wd=1e-6_opt=SGD_sched=CosAnn_reset-opt=True_lamb1=0.05_lamb2=0.95'
    # checkpoint = 'model_10000.tar'
    # model = '24_PLOT-GRAD-ANGLES_SUM-LOSSES_mica_duo_MULTITASK_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_lr=1e-5_arc-lr=1e-5_fr-lr=1e-5_wd=2e-5_lamb1=0.02_lamb2=0.98'
    # checkpoint = 'model_30000.tar'    # LFW: 90.4%,  MLFW: 70.7%,   TALFW: 64.5%
    # checkpoint = 'model_40000.tar'    # LFW: 91.3%,  MLFW: 71.0%,   TALFW: 65.4%
    # checkpoint = 'model_100000.tar'   # LFW: 90.7%,  MLFW: 66.3%,   TALFW: 70.2%
    # checkpoint = 'model_200000.tar'   # LFW: 90.4%,  MLFW: 62.4%,   TALFW: 74.3%
    # model = '24_PLOT-GRAD-ANGLES_SUM-LOSSES_mica_duo_MULTITASK_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_lr=1e-5_arc-lr=1e-5_fr-lr=1e-5_wd=2e-5_lamb1=0.01_lamb2=0.99'
    # checkpoint = 'model_30000.tar'    # LFW: 89.5%,  MLFW: 70.3%,   TALFW: 63.7%
    # checkpoint = 'model_100000.tar'   # LFW: 90.8%,  MLFW: 68.0%,   TALFW: 68.0%
    # model = '26_SANITY-CHECK_MULTI-TASK_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_loss=arcface_marg1=0.5_scal1=32_wd=1e-5_opt=SGD_sched=CosAnn_reset-opt=True_lamb1=1.0_lamb2=1.0'
    # checkpoint = 'model.tar'          # LFW: 96.9%,  MLFW: 76.0%,   TALFW: 68.3%
    # checkpoint = 'model_10000.tar'    # LFW: 97.4%,  MLFW: 77.2%,   TALFW: 69.8%
    # checkpoint = 'model_150000.tar'   # LFW: 97.4%,  MLFW: 73.3%,   TALFW: 74.4%
    # checkpoint = 'model_200000.tar'   # LFW: 97.0%,  MLFW: 71.9%,   TALFW: 73.6%

    # Separated Multi-task (3DMM + Reconstruction)
    # model = '21_TRAIN-TASK-SEPARATED_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_lr=1e-5_arc-lr=1e-5_fr-lr=1e-5_wd=1e-6_opt=SGD_sched=CosAnn_reset-opt=True_lamb1=1.0_lamb2=1.0'
    # checkpoint = 'model_40000.tar'
    # model = '21_TRAIN-TASK-SEPARATED_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_lr=1e-5_arc-lr=1e-5_fr-lr=1e-5_wd=1e-6_opt=SGD_sched=CosAnn_reset-opt=True_lamb1=0.05_lamb2=0.95'
    # checkpoint = 'model_10000.tar'

    # FUSION 2D + 3D
    # model = '17_mica_duo_MULTITASK-ARCFACE-ACC-CONFMAT-FUSION_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=arcface-3dmm_lr=1e-5_arc-lr=1e-5_fr-lr=1e-7_wd=2e-5_lamb1=0.0_lamb2=1.0'
    # checkpoint = 'model_230000.tar'
    # model = '17_mica_duo_MULTITASK-ARCFACE-ACC-CONFMAT-FUSION_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=arcface-3dmm_lr=1e-5_arc-lr=1e-5_fr-lr=1e-7_wd=2e-5_lamb1=1.0_lamb2=1.0'
    # checkpoint = 'model_20000.tar'

    # MULT-task FUSION (2D + 3D)
    # model = '17_mica_duo_MULTITASK-ARCFACE-ACC-CONFMAT-FUSION_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=arcface-3dmm_lr=1e-5_arc-lr=1e-5_fr-lr=1e-7_wd=2e-5_lamb1=1.0_lamb2=1.0'
    # checkpoint = 'model_20000.tar'
    # model = '17_mica_duo_MULTITASK-ARCFACE-ACC-CONFMAT-FUSION_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_eval=20perc_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=arcface-3dmm_lr=1e-5_arc-lr=1e-5_fr-lr=1e-8_wd=2e-5_lamb1=1.0_lamb2=1.0'
    # checkpoint = 'model_20000.tar'

    # BERNARDO
    if not '--cfg' in sys.argv:
        sys.argv.append('--cfg')
        sys.argv.append(configs_folder + '/' + model + '.yml')
    
    if not '--checkpoint' in sys.argv:
        sys.argv.append('--checkpoint')
        sys.argv.append(models_folder + '/' + model + '/' + checkpoint)

    if not '--test_dataset' in sys.argv:
        sys.argv.append('--test_dataset')
        sys.argv.append('LFW')
        # sys.argv.append('MLFW')
        # sys.argv.append('TALFW')


    cfg, args = parse_args()

    if cfg.cfg_file is not None:
        # exp_name = cfg.cfg_file.split('/')[-1].split('.')[0]              # original
        exp_name = '.'.join(cfg.cfg_file.split('/')[-1].split('.')[:-1])    # Bernardo
        # print('test_multitask_facerecognition1 - __main__ - cfg.cfg_file:', cfg.cfg_file)
        cfg.output_dir = os.path.join('./output', exp_name)

    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.empty_cache()
    num_gpus = torch.cuda.device_count()

    # BERNARDO
    if num_gpus == 0:
        num_gpus = 1    # cpu

    # mp.spawn(test, args=(num_gpus, cfg, args), nprocs=num_gpus, join=True)                            # original
    mp.spawn(test_multitask_facerecognition1, args=(num_gpus, cfg, args), nprocs=num_gpus, join=True)   # Bernardo

    exit(0)
