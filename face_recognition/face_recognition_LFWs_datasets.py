import argparse
import sys
import os
import random
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from yacs.config import CfgNode as CN

from file_tree import FileTreeLfwDatasets, FileTreeLfwDatasets3dReconstructed


def deterministic(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    np.random.seed(rank)
    random.seed(rank)
    cudnn.deterministic = True
    cudnn.benchmark = False


def main_face_recognition(cfg: CN):
    print('__main__(): main_face_recognition(): reading LFWs RGB images... ', end='')
    common_subjects, common_file_names = FileTreeLfwDatasets().get_common_images_names_without_ext(
                                                                                            cfg.input_rgb_images_path,
                                                                                            cfg.datasets_names,
                                                                                            cfg.file_exts,
                                                                                            'original')
    print('finished!', 'common_subjects=' + str(len(common_subjects)) + '  common_file_names='+str(len(common_file_names)))

    print('__main__(): main_face_recognition(): reading LFWs 3D reconstructed faces... ', end='')
    reconst_faces_file_names, reconst_render_file_names = FileTreeLfwDatasets3dReconstructed().get_3d_faces_file_names(
                                                                                    cfg.input_3d_reconstructions_path,
                                                                                    cfg.datasets_names[0],
                                                                                    common_subjects,
                                                                                    common_file_names)
    print('finished!')




if __name__ == '__main__':
    cfg = CN()

    cfg.device = 'cuda:0'  # original
    # cfg.device = 'cuda:1'    # BERNARDO

    cfg.input_rgb_images_path = '/mnt/42C8A18221CA5B0F/Local/datasets/imagensRGB'
    # cfg.input_rgb_images_path = 'sftp://duo/home/bjgbiesseck/datasets/rgb_images'
    # cfg.input_rgb_images_path = '/home/bjgbiesseck/datasets/rgb_images'          # duo

    cfg.input_3d_reconstructions_path = '/home/biesseck/GitHub/MICA/demo/output'
    # cfg.input_3d_reconstructions_path = '/home/biesseck/GitHub_duo/MICA/demo/output'
    # cfg.input_3d_reconstructions_path = '/home/bjgbiesseck/GitHub/MICA/demo/output'  # duo

    cfg.datasets_names = ['lfw', 'TALFW']
    # cfg.datasets_names = ['calfw', 'MLFW']

    cfg.file_exts = ['jpg', 'png']

    deterministic(440)
    # main_face_recognition(args)

    main_face_recognition(cfg)
