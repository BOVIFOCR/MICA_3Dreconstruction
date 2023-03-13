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


import os, sys
from glob import glob

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align
from loguru import logger
from pytorch3d.io import save_ply
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp
from tqdm import tqdm

from configs.config import cfg
from utils import util

# Bernardo
from face_recognition.dataloaders.mlfw_verif_pairs_imgs import MLFW_Verif_Pairs_Images
from face_recognition.dataloaders.lfw_verif_pairs_imgs import LFW_Verif_Pairs_Images

input_mean = 127.5
input_std = 127.5

# NOW_SCANS = '/home/wzielonka/datasets/NoWDataset/final_release_version/scans/'                # original
# NOW_PICTURES = '/home/wzielonka/datasets/NoWDataset/final_release_version/iphone_pictures/'   # original
# NOW_BBOX = '/home/wzielonka/datasets/NoWDataset/final_release_version/detected_face/'         # original
# STIRLING_PICTURES = '/home/wzielonka/datasets/Stirling/images/'                               # original
# NOW_SCANS = '/datasets1/bjgbiesseck/NoWDataset/NoW_Dataset/final_release_version/scans/'                  # Bernardo
# NOW_PICTURES = '/datasets1/bjgbiesseck/NoWDataset/NoW_Dataset/final_release_version/iphone_pictures/'     # Bernardo
# NOW_BBOX = '/datasets1/bjgbiesseck/NoWDataset/NoW_Dataset/final_release_version/detected_face/'           # Bernardo
# STIRLING_PICTURES = '/datasets1/bjgbiesseck/Stirling/images/'                                             # Bernardo

# MLFW_PICTURES = '/datasets1/bjgbiesseck/MLFW/origin'
MLFW_PICTURES = '/datasets1/bjgbiesseck/MLFW/aligned'
MLFW_VERIF_PAIRS_LIST = '/datasets1/bjgbiesseck/MLFW/pairs.txt'

# LFW_PICTURES = '/datasets1/bjgbiesseck/lfw'
LFW_PICTURES = '/datasets1/bjgbiesseck/lfw_cropped_aligned'
LFW_VERIF_PAIRS_LIST = '/datasets1/bjgbiesseck/lfw_cropped_aligned/pairs.txt'

class TesterMultitaskFacerverification(object):
    def __init__(self, nfc_model, config=None, device=None):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config

        self.device = device
        self.batch_size = self.cfg.dataset.batch_size
        self.K = self.cfg.dataset.K
        self.render_mesh = True
        self.embeddings_lyhm = {}

        # deca model
        self.nfc = nfc_model.to(self.device)
        self.nfc.testing = True

        logger.info(f'[INFO]            {torch.cuda.get_device_name(device)}')

    def load_checkpoint(self, model_path):
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.device}

        checkpoint = torch.load(model_path, map_location)

        if 'arcface' in checkpoint:
            self.nfc.arcface.load_state_dict(checkpoint['arcface'])
        if 'flameModel' in checkpoint:
            self.nfc.flameModel.load_state_dict(checkpoint['flameModel'])
        if 'faceClassifier' in checkpoint:                                         # Bernardo
            self.nfc.faceClassifier.load_state_dict(checkpoint['faceClassifier'])  # Bernardo

        logger.info(f"[TESTER] Resume from {model_path}")

    def load_model_dict(self, model_dict):
        dist.barrier()

        self.nfc.canonicalModel.load_state_dict(model_dict['canonicalModel'])
        self.nfc.arcface.load_state_dict(model_dict['arcface'])

    # Bernardo
    def process_image(self, img, app):
        # images = []
        
        # Bernardo
        if not app is None:
            bboxes, kpss = app.det_model.detect(img, max_num=0, metric='default')
            if bboxes.shape[0] != 1:
                logger.error('Face not detected!')
                return images
            i = 0
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            aimg = face_align.norm_crop(img, landmark=face.kps)
        else:
            aimg = img

        # blob = cv2.dnn.blobFromImages([aimg], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=True)
        blob = cv2.dnn.blobFromImages([aimg], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=False)

        # images.append(torch.tensor(blob[0])[None])   # original
        # return images                                # original
        image = torch.tensor(blob[0])[None]            # Bernardo
        return image                                   # Bernardo


    def get_name(self, best_model, id):
        if '_' in best_model:
            name = id if id is not None else best_model.split('_')[-1][0:-4]
        else:
            name = id if id is not None else best_model.split('/')[-1][0:-4]
        return name


    def test_mlfw(self, best_model, id=None):
        self.load_checkpoint(best_model)
        name = self.get_name(best_model, id)
        self.mlfw(name)


    def test_lfw(self, best_model, id=None):
        self.load_checkpoint(best_model)
        name = self.get_name(best_model, id)
        self.lfw(name)


    def save_mesh(self, file, vertices):
        scaled = vertices * 1000.0
        save_ply(file, scaled.cpu(), self.nfc.render.faces[0].cpu())

        # mask = self.nfc.masking.get_triangle_whole_mask()
        # v, f = self.nfc.masking.get_masked_mesh(vertices, mask)
        # save_obj(file, v[0], f[0])


    # Bernardo
    def cache_to_cuda(self, cache):
        for key in cache.keys():
            # i, a = cache[key]                                                        # original
            # cache[key] = (i.to(self.device), a.to(self.device))                      # original
            i0, i1, l = cache[key]                                                     # Bernardo
            cache[key] = (i0.to(self.device), i1.to(self.device), l.to(self.device))   # Bernardo
        return cache


    # Bernardo
    def create_mlfw_cache(self):
        cache_file_name = 'test_mlfw_cache.pt'      # Bernardo
        cache = {}

        file_ext = '.jpg'
        all_pairs, pos_pair_label, neg_pair_label = MLFW_Verif_Pairs_Images().load_pairs_samples_protocol_from_file(MLFW_VERIF_PAIRS_LIST, MLFW_PICTURES, file_ext)

        arcface = []
        for i, pair in enumerate(all_pairs):
            path_img0, path_img1, label_pair = pair
            print('\x1b[2K', end='')
            print(f'tester_multitask_FACEVERIFICATION - create_mlfw_cache - {i}/{len(all_pairs)-1} ({path_img0}, {path_img1}, {label_pair})', end='\r')
            img0 = imread(path_img0)[:, :, :3]
            img1 = imread(path_img1)[:, :, :3]
            
            img0_preproc = self.process_image(img0.astype(np.float32), app=None)
            img1_preproc = self.process_image(img1.astype(np.float32), app=None)
            
            cache[i] = (img0_preproc, img1_preproc, torch.tensor(int(label_pair)))

        # torch.save(cache, cache_file_name)         # Bernardo
        return self.cache_to_cuda(cache)

    

    # Bernardo
    def create_lfw_cache(self, output_folder='lfw_cropped_aligned'):
        cache_file_name = 'test_lfw_cache.pt'      # Bernardo
        cache = {}

        file_ext = '.png'
        all_pairs, pos_pair_label, neg_pair_label = LFW_Verif_Pairs_Images().load_pairs_samples_protocol_from_file(LFW_VERIF_PAIRS_LIST, LFW_PICTURES, file_ext)

        arcface = []
        for i, pair in enumerate(all_pairs):
            path_img0, path_img1, label_pair = pair
            print('\x1b[2K', end='')
            print(f'tester_multitask_FACEVERIFICATION - create_lfw_cache - {i}/{len(all_pairs)-1} ({path_img0}, {path_img1}, {label_pair})', end='\r')
            img0 = imread(path_img0)[:, :, :3]
            img1 = imread(path_img1)[:, :, :3]
            
            img0_preproc = self.process_image(img0.astype(np.float32), app=None)
            img1_preproc = self.process_image(img1.astype(np.float32), app=None)
            
            cache[i] = (img0_preproc, img1_preproc, torch.tensor(int(label_pair)))

        # torch.save(cache, cache_file_name)         # Bernardo
        return self.cache_to_cuda(cache)


    # Bernardo
    def get_all_distances(self, cache):
        cache_keys = list(cache.keys())
        cos_sims = torch.zeros(len(cache_keys))

        for i, key in enumerate(cache_keys):
            img0, img1, pair_label = cache[key]
            with torch.no_grad():
                codedict = self.nfc.encode(images=None, arcface_imgs=torch.cat([img0, img1], dim=0))
                opdict = self.nfc.decode(codedict, 0)

                face_embedd = opdict['face_embedd']
                # face_embedd = opdict['logits_pred']

                cos_sims[i] = F.cosine_similarity(F.normalize(torch.unsqueeze(face_embedd[0], 0)), F.normalize(torch.unsqueeze(face_embedd[1], 0)))
                print('\x1b[2K', end='')
                print(f'tester_multitask_FACEVERIFICATION - get_all_distances - {i}/{len(cache_keys)-1} - pair_label: {pair_label}, cos_sims[{i}]: {cos_sims[i]}', end='\r')

        return cos_sims
    

    # Bernardo
    def find_best_treshold(self, cache, cos_sims):
        best_tresh = 0
        best_acc = 0
        
        start, end, step = 0, 1, 0.01
        treshs = torch.arange(start, end+step, step)
        for i, tresh in enumerate(treshs):
            tresh = torch.round(tresh, decimals=3)
            tp, fp, tn, fn, acc = 0, 0, 0, 0, 0
            for i, cos_sim in enumerate(cos_sims):
                _, _, pair_label = cache[i]
                if pair_label == 1:
                    if cos_sim >= tresh:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if cos_sim < tresh:
                        tn += 1
                    else:
                        fp += 1

            acc = round((tp + tn) / (tp + tn + fp + fn), 4)
            # print(f'tester_multitask_FACEVERIFICATION - {i}/{treshs.size()[0]-1} - tresh: {tresh} - acc: {acc}')

            if acc > best_acc:
                best_acc = acc
                best_tresh = tresh
            
            print('\x1b[2K', end='')
            print(f'tester_multitask_FACEVERIFICATION - {i}/{len(treshs)-1} - tresh: {tresh}', end='\r')

        return best_tresh, best_acc


    # Bernardo
    def evaluate_model(self, best_id, dataset_name):
        logger.info(f"[TESTER] {dataset_name} testing has begun!")
        self.nfc.eval()

        logger.info(f"[TESTER] Creating {dataset_name} cache...")
        # cache = self.create_now_cache()       # original
        if dataset_name.upper() == 'MLFW':
            cache = self.create_mlfw_cache()    # Bernardo
        elif dataset_name.upper() == 'LFW':
            cache = self.create_lfw_cache()     # Bernardo
        else:
            logger.error('[TESTER] Test dataset was not specified: ' + str(dataset_name))
            sys.exit(0)

        logger.info(f"Computing pair distances...")
        cos_sims = self.get_all_distances(cache)

        print()
        logger.info(f"Findind best treshold...")
        best_tresh, best_acc = self.find_best_treshold(cache, cos_sims)
        print(f'\nbest_tresh: {best_tresh},   best_acc: {best_acc}')
