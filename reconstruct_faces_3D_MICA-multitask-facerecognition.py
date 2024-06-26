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


import argparse
import os, sys
import random
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align
from loguru import logger
from pytorch3d.io import save_ply, save_obj
from skimage.io import imread
from tqdm import tqdm

# from configs.config import get_cfg_defaults                           # original
from configs.config_multitask_facerecognition import get_cfg_defaults   # Bernardo

from datasets.creation.util import get_arcface_input, get_center
from utils import util


def deterministic(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    np.random.seed(rank)
    random.seed(rank)

    cudnn.deterministic = True
    cudnn.benchmark = False


# BERNARDO
def process_BERNARDO(args, app, image_size=224):
    dst = Path(args.a)
    print('process_BERNARDO - dst:', dst)
    dst.mkdir(parents=True, exist_ok=True)
    processes = []
    # image_paths = sorted(glob(args.i + '/*.*'))                                     # original
    image_paths = sorted(glob(args.i + '/*.jpg')) + sorted(glob(args.i + '/*.png'))   # BERNARDO

    # print('process_BERNARDO - image_paths:', image_paths)
    # sys.exit(0)

    for image_path in tqdm(image_paths):
        name = Path(image_path).stem
        img = cv2.imread(image_path)
        # print('demo.py: process_BERNARDO - image_path=', image_path)
        bboxes, kpss = app.det_model.detect(img, max_num=0, metric='default')
        if bboxes.shape[0] == 0:
            continue
        i = get_center(bboxes, img)
        bbox = bboxes[i, 0:4]
        det_score = bboxes[i, 4]
        kps = None
        if kpss is not None:
            kps = kpss[i]
        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        blob, aimg = get_arcface_input(face, img)
        file = str(Path(dst, name))
        np.save(file, blob)
        cv2.imwrite(file + '.jpg', face_align.norm_crop(img, landmark=face.kps, image_size=image_size))
        processes.append(file + '.npy')

    return processes



def to_batch(path):
    src = path.replace('.npy', '.jpg')
    if not os.path.exists(src):
        src = path.replace('.npy', '.png')

    image = imread(src)[:, :, :3]
    image = image / 255.
    image = cv2.resize(image, (224, 224)).transpose(2, 0, 1)
    image = torch.tensor(image).cuda()[None]

    arcface = np.load(path)
    arcface = torch.tensor(arcface).cuda()[None]

    return image, arcface


def load_checkpoint(args, mica):
    checkpoint = torch.load(args.m)
    if 'arcface' in checkpoint:
        mica.arcface.load_state_dict(checkpoint['arcface'])
    if 'flameModel' in checkpoint:
        mica.flameModel.load_state_dict(checkpoint['flameModel'])


# BERNARDO
class Tree:
    def walk(self, dir_path: Path):
        contents = list(dir_path.iterdir())
        for path in contents:
            if path.is_dir():  # extend the prefix and recurse:
                yield str(path)
                yield from self.walk(path)

    def get_all_sub_folders(self, dir_path: str):
        folders = [dir_path]
        for folder in Tree().walk(Path(os.getcwd()) / dir_path):
            # print(folder)
            folders.append(folder)
        return sorted(folders)



def main(cfg, args):
    device = 'cuda:0'  # original
    # device = 'cuda:1'    # BERNARDO

    cfg.model.testing = True
    mica = util.find_model_using_name(model_dir='micalib.models', model_name=cfg.model.name)(cfg, device)
    load_checkpoint(args, mica)
    mica.eval()

    # Bernardo
    arcface_exp_folder = args.a
    output_exp_folder = args.o

    faces = mica.render.faces[0].cpu()
    Path(args.i.replace('input', arcface_exp_folder)).mkdir(exist_ok=True, parents=True)
    Path(args.i.replace('input', output_exp_folder)).mkdir(exist_ok=True, parents=True)

    app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(224, 224))

    with torch.no_grad():
        logger.info(f'Processing has started...')

        # LIST ALL DIRS (BERNARDO)
        print('Loading folders names...')
        sub_folders = Tree().get_all_sub_folders(args.i)
        # print('sub_folders:', sub_folders)
        begin_index = 0
        end_index = len(sub_folders)

        if args.str_begin != '':
            print('Searching str_begin \'' + args.str_begin + '\' ...  ')
            for x, sub_folder in enumerate(sub_folders):
                if args.str_begin in sub_folder:
                    begin_index = x
                    print('found at', begin_index)
                    break

        if args.str_end != '':
            print('Searching str_end \'' + args.str_end + '\' ...  ')
            for x, sub_folder in enumerate(sub_folders):
                if args.str_end in sub_folder:
                    end_index = x+1
                    print('found at', begin_index)
                    break

        print('\n------------------------')
        print('begin_index:', begin_index)
        print('end_index:', end_index)
        print('------------------------\n')

        for args.i in sub_folders[begin_index:end_index]:
            args.a = args.i.replace('input', arcface_exp_folder)
            args.o = args.i.replace('input', output_exp_folder)

            # paths = process(args, app)   # original
            paths = process_BERNARDO(args, app)   # BERNARDO

            for path in tqdm(paths):
                print('path:', path)
                name = Path(path).stem
                images, arcface = to_batch(path)
                codedict = mica.encode(images, arcface)
                opdict = mica.decode(codedict)
                meshes = opdict['pred_canonical_shape_vertices']
                code = opdict['pred_shape_code']
                lmk = mica.flame.compute_landmarks(meshes)

                mesh = meshes[0]
                landmark_51 = lmk[0, 17:]
                landmark_7 = landmark_51[[19, 22, 25, 28, 16, 31, 37]]
                rendering = mica.render.render_mesh(mesh[None])
                image = (rendering[0].cpu().numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                image = np.minimum(np.maximum(image, 0), 255).astype(np.uint8)

                dst = Path(args.o, name)
                dst.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(f'{dst}/render.jpg', image)
                save_ply(f'{dst}/mesh.ply', verts=mesh.cpu() * 1000.0, faces=faces)  # save in millimeters
                save_obj(f'{dst}/mesh.obj', verts=mesh.cpu() * 1000.0, faces=faces)
                np.save(f'{dst}/identity', code[0].cpu().numpy())
                np.save(f'{dst}/kpt7', landmark_7.cpu().numpy() * 1000.0)
                np.save(f'{dst}/kpt68', lmk.cpu().numpy() * 1000.0)

            logger.info(f'Processing finished. Results has been saved in {args.o}')
            print(f'Processing finished. Results has been saved in {args.o}')
            print('------------------------------')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MICA - Towards Metrical Reconstruction of Human Faces')
    # parser.add_argument('-i', default='demo/input', type=str, help='Input folder with images')                                                # original
    # parser.add_argument('-i', default='demo/input_TESTE', type=str, help='Input folder with images')                                          # BERNARDO
    # parser.add_argument('-i', default='demo/input/lfw', type=str, help='Input folder with images')                                            # BERNARDO
    # parser.add_argument('-i', default='demo/input/CelebA/Img/img_align_celeba', type=str, help='Input folder with images')                    # BERNARDO
    parser.add_argument('-i', default='demo/input/MS-Celeb-1M/ms1m-retinaface-t1/images', type=str, help='Input folder with images')          # BERNARDO
    # parser.add_argument('-i', default='demo/input/MS-Celeb-1M/ms1m-retinaface-t1/images_reduced', type=str, help='Input folder with images')  # BERNARDO
    # parser.add_argument('-i', default='demo/input/MLFW_small', type=str, help='Input folder with images')                                     # BERNARDO
    # parser.add_argument('-i', default='demo/input/MLFW/origin', type=str, help='Input folder with images')                                      # BERNARDO

    # parser.add_argument('-exp', default='', type=str, help='Processed images for MICA input')
    # parser.add_argument('-exp', default='4_mica_duo_TESTS_train=FRGC,LYHM,FLORENCE,FACEWAREHOUSE_eval=Stirling_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100', type=str, help='Processed images for MICA input')
    # parser.add_argument('-exp', default='6_mica_duo_TESTS_train=FRGC,LYHM,FLORENCE,FACEWAREHOUSE_eval=Stirling_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100', type=str, help='Processed images for MICA input')
    # parser.add_argument('-exp', default='11_mica_duo_MULTITASK-VALIDATION-WORKING_train=FRGC,LYHM,FLORENCE,FACEWAREHOUSE_eval=Stirling_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_lamb1=0.5_lamb2=1.0', type=str, help='Processed images for MICA input')
    # parser.add_argument('-exp', default='11_mica_duo_MULTITASK-VALIDATION-WORKING_train=FRGC,LYHM,FLORENCE,FACEWAREHOUSE_eval=Stirling_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_lamb1=0.5_lamb2=1.0', type=str, help='Processed images for MICA input')
    # parser.add_argument('-exp', default='12_mica_duo_MULTITASK-VALIDATION-WORKING_train=FRGC,LYHM,Stirling,FACEWAREHOUSE_eval=FLORENCE_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-6_lamb1=0.5_lamb2=1.0', type=str, help='Processed images for MICA input')
    # parser.add_argument('-exp', default='12_mica_duo_MULTITASK-VALIDATION-WORKING_train=FRGC,LYHM,Stirling,FACEWAREHOUSE_eval=FLORENCE_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_lamb1=0.5_lamb2=1.0', type=str, help='Processed images for MICA input')
    # parser.add_argument('-exp', default='12_mica_duo_MULTITASK-VALIDATION-WORKING_train=FRGC,LYHM,Stirling,FACEWAREHOUSE_eval=FLORENCE_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_lamb1=0.5_lamb2=1.0', type=str, help='Processed images for MICA input')

    parser.add_argument('-exp', default='26_SANITY-CHECK_MULTI-TASK_train=FRGC,LYHM,Stirling,FACEWAREHOUSE,FLORENCE_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_loss=arcface_marg1=0.5_scal1=32_wd=1e-5_opt=SGD_sched=CosAnn_reset-opt=True_lamb1=1.0_lamb2=1.0', type=str, help='Processed images for MICA input')

    # parser.add_argument('-o', default='output' +'_'+exp, type=str, help='Output folder')
    # parser.add_argument('-a', default='arcface'+'_'+exp, type=str, help='Processed images for MICA input')

    # parser.add_argument('-m', default='data/pretrained/mica.tar', type=str, help='Pretrained model path')    # original
    # parser.add_argument('-m', default='/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/output/'+exp+'/best_models/best_model_3.tar', type=str, help='Pretrained model path')      # Bernardo
    # parser.add_argument('-m', default='/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/output/'+exp+'/model.tar', type=str, help='Pretrained model path')      # Bernardo

    parser.add_argument('-str_begin', default='', type=str, help='Substring to find and start processing')
    parser.add_argument('-str_end', default='', type=str, help='Substring to find and stop processing')

    args = parser.parse_args()
    
    # args.o = 'output'  + '_' + args.exp
    args.o = 'output/MS-Celeb-1M/ms1m-retinaface-t1/3D_reconstruction/' + args.exp + '/'
    
    args.a = 'arcface' + '_' + args.exp
    
    # args.m = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/output/' + args.exp + '/model.tar'
    args.m = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/output/' + args.exp + '/model_290000.tar'

    cfg = get_cfg_defaults()

    deterministic(42)
    main(cfg, args)
