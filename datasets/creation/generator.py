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

import sys
import os
from glob import glob
from multiprocessing import Pool
from pathlib import Path
from typing import List

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align
from loguru import logger
from tqdm import tqdm
import shutil

# from datasets.creation.instances.instance import Instance                    # Original
# from datasets.creation.util import get_image, get_center, get_arcface_input  # Original
from instances.instance import Instance                                        # BERNARDO
from util import get_image, get_center, get_arcface_input                      # BERNARDO


def _transfer(src, dst):
    src.parent.mkdir(parents=True, exist_ok=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.system(f'cp {str(src)} {str(dst)}')


def _copy(payload):
    instance, func, target, transform_path = payload
    files = func()
    for actor in files.keys():
        for file in files[actor]:
            _transfer(Path(file), Path(instance.get_dst(), target, actor, transform_path(file)))


class Generator:
    def __init__(self, instances):
        self.instances: List[Instance] = instances
        self.ARCFACE = '_arcface_input'

    def copy(self):
        tqdm.write('Start copying...')
        for instance in tqdm(self.instances):
            payloads = [(instance, instance.get_images, 'images', instance.transform_path)]
            
            # BERNARDO
            tqdm.write('payloads: ' + str(payloads))
            tqdm.write('instance: ' + str(instance))
            
            with Pool(processes=len(payloads)) as pool:
                for _ in tqdm(pool.imap_unordered(_copy, payloads), total=len(payloads)):
                    pass

    def preprocess(self):
        logger.info('Start preprocessing...')
        for instance in tqdm(self.instances):
            instance.preprocess()

    def arcface(self):
        app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(224, 224))

        logger.info('Start arcface...')

        # for instance in tqdm(self.instances):    # original
        for instance in self.instances:            # Bernardo
            
            # src = instance.get_dst()             # original
            src = instance.get_src()               # Bernardo
            # img_ext = instance.get_img_ext()     # Bernardo
            # image_paths = sorted(glob(f'{src}*{img_ext}'))

            tqdm.write('src: ' + src)
            tqdm.write('path: ' + f'{src}*')
            # tqdm.write('image_paths:' + str(image_paths))

            images_without_faces = []

            for image_path in tqdm(sorted(glob(f'{src}/images/*/*'))):
                image_path = image_path.replace('//', '/')    # Bernardo
                tqdm.write('image_path: ' + str(image_path))  # Bernardo
                
                dst = image_path.replace('images', self.ARCFACE)
                assert src != dst    # Bernardo
                
                tqdm.write('dst: ' + str(dst))    # Bernardo
                
                Path(dst).parent.mkdir(exist_ok=True, parents=True)
                for img in instance.transform_image(get_image(image_path[0:-4])):
                    bboxes, kpss = app.det_model.detect(img, max_num=0, metric='default')

                    # # Bernardo
                    # if 'M1001/M1001_004.jpg' in image_path:
                    #     tqdm.write('img: ' + str(img))  # Bernardo
                    #     tqdm.write('bboxes: ' + str(bboxes))  # Bernardo
                    #     tqdm.write('kpss: ' + str(kpss))  # Bernardo
                    #     input('PAUSED')

                    if bboxes.shape[0] == 0:
                        # tqdm.write('Face not found - image_path: ' + image_path)    # Bernardo
                        # sys.exit(0)
                        images_without_faces.append(image_path)
                        continue

                    i = get_center(bboxes, img)
                    bbox = bboxes[i, 0:4]
                    det_score = bboxes[i, 4]
                    if det_score < instance.get_min_det_score():
                        continue
                    kps = None
                    if kpss is not None:
                        kps = kpss[i]
                    face = Face(bbox=bbox, kps=kps, det_score=det_score)
                    blob, aimg = get_arcface_input(face, img)
                    np.save(dst[0:-4], blob)
                    cv2.imwrite(dst, face_align.norm_crop(img, landmark=face.kps, image_size=224))

                # input('PAUSED')   # Bernardo
                # sys.exit(0)       # Bernardo


                # BERNARDO: COPY ".npz" FILES (FLAME PARAMETERS)
                subj = image_path.split('/')[-2]
                dir_src_npz_file = '/'.join(image_path.split('/')[:-3]) + '/FLAME_parameters/' + subj
                npz_pattern_to_search = dir_src_npz_file + '/' + '*.npz'
                # tqdm.write('dir_src_npz_file: ' + str(dir_src_npz_file))
                found_file = glob(npz_pattern_to_search, recursive=True)

                if len(found_file) == 0:
                    print('Error, file not found:', file_name_to_search)
                    sys.exit(0)
                elif len(found_file) > 1:
                    print('Error, multiple files found:', found_file)
                    sys.exit(0)

                found_file = found_file[0]
                tqdm.write('found_file: ' + str(found_file))
                output_npz_file = '/'.join(dst.split('/')[0:-1]) + '/' + found_file.split('/')[-1]
                tqdm.write('output_npz_file: ' + str(output_npz_file))
                assert found_file != output_npz_file
                shutil.copyfile(found_file, output_npz_file)
                # BERNARDO: COPY ".npz" FILES (FLAME PARAMETERS)

                # input('PAUSED')   # Bernardo
                # sys.exit(0)       # Bernardo

            if len(images_without_faces) > 0:
                tqdm.write('\n\nIMAGES WITHOUT FACES:')
                for path in images_without_faces:
                    tqdm.write('path: ' + str(path))



    def run(self):
        self.copy()
        self.preprocess()
        self.arcface()
