import os
import sys

import argparse
import random
import socket
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pytorch3d.io import load_obj
from pytorch3d.loss import chamfer_distance
from mpl_toolkits.mplot3d import Axes3D

import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.io import load_obj, load_ply
from pytorch3d.loss import chamfer_distance

from models.arcface import Arcface


input_mean = 127.5
input_std = 127.5


def get_parts_indices(sub_folders, divisions):
    begin_div = []
    end_div = []
    div_size = int(len(sub_folders) / divisions)
    remainder = int(len(sub_folders) % divisions)

    for i in range(0, divisions):
        begin_div.append(i*div_size)
        end_div.append(i*div_size + div_size)
    
    end_div[-1] += remainder

    # print('begin_div:', begin_div)
    # print('end_div:', end_div)
    # sys.exit(0)
    return begin_div, end_div


def load_sample(file_path):
    if file_path.endswith('.obj'):
        verts, _ = load_obj(file_path)
        vertices = verts.verts_packed()
    elif file_path.endswith('.ply'):
        data = load_ply(file_path)
        # vertices = data['vertices']
        vertices = data[0]
    elif file_path.endswith('.npy'):
        vertices = np.load(file_path)
        vertices = torch.from_numpy(vertices)
    elif file_path.endswith('.jpg') or file_path.endswith('.png'):
        vertices_bgr = cv2.imread(file_path)
        vertices_blob = cv2.dnn.blobFromImages([vertices_bgr], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=True)
        vertices = torch.from_numpy(vertices_blob)
        if len(vertices.shape) > 3:
            vertices = torch.squeeze(vertices)
    else:
        raise ValueError("Unsupported file format. Only \'.obj\', \'.ply\', \'.npy\', \'.jpg\', and \'.png\' files are supported.")
    return vertices
    

def compute_chamfer_distance(points1, points2):
    chamfer_dist = chamfer_distance(points1.unsqueeze(0), points2.unsqueeze(0))
    return chamfer_dist[0]


def compute_cosine_distance(array1, array2, normalize=True):
    # print('array1.shape:', array1.shape)
    if normalize == True:
        array1 = torch.nn.functional.normalize(array1, dim=0)
        array2 = torch.nn.functional.normalize(array2, dim=0)
    cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)(array1, array2)
    cos_dist = 1.0 - cos_sim
    # print('\ncos_sim:', cos_sim)
    # sys.exit(0)
    return cos_sim


def compute_euclidean_distance(array1, array2, normalize=True):
    # print('array1.shape:', array1.shape)
    if normalize == True:
        array1 = torch.nn.functional.normalize(array1, dim=0)
        array2 = torch.nn.functional.normalize(array2, dim=0)
    eucl_dist = torch.norm(array1 - array2)
    return eucl_dist


def main(args):
    assert args.part < args.divs, f'Error, args.part ({args.part}) >= args.divs ({args.divs}), but should be args.part ({args.part}) < args.divs ({args.divs})'

    r100_arcface_path = 'models/models_weights/arcface-torch/ms1mv3_arcface_r100_fp16/backbone.pth'
    device = 'cuda:0'
    print(f'Loading face recognition model from \'{r100_arcface_path}\'')
    arcface = Arcface(pretrained_path=r100_arcface_path).to(device)
    arcface.eval()

    dataset_path = args.input_path.rstrip('/')
    output_path = os.path.join(os.path.dirname(dataset_path), 'embeddings2D')
    os.makedirs(output_path, exist_ok=True)

    print('dataset_path:', dataset_path)
    print('Searching subject subfolders...')
    subjects_paths = sorted([os.path.join(dataset_path,subj) for subj in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, subj))])
    # print('subjects_paths:', subjects_paths)
    print(f'Found {len(subjects_paths)} subjects!')

    begin_parts, end_parts = get_parts_indices(subjects_paths, args.divs)
    idx_subj_begin, idx_subj_end = begin_parts[args.part], end_parts[args.part]
    num_subjs_part = idx_subj_end - idx_subj_begin 
    print('\nbegin_parts:', begin_parts)
    print('end_parts:  ', end_parts)
    print(f'idx_subj_begin: {idx_subj_begin}    idx_subj_end: {idx_subj_end}')
    print('')
    # sub_folders = subjects_paths[begin_parts[args.part]:end_parts[args.part]]

    # print('Computing 2D face embeddings...\n')
    for idx_subj, subj_path in enumerate(subjects_paths):
        if idx_subj >= idx_subj_begin and idx_subj < idx_subj_end:
            subj_start_time = time.time()

            subj = os.path.basename(subj_path)
            output_subj_path = os.path.join(output_path, subj)
            os.makedirs(output_subj_path, exist_ok=True)

            print(f'subj {idx_subj-idx_subj_begin}/{num_subjs_part}: {subj} - Searching samples...')
            samples_file_name = sorted([sample for sample in os.listdir(subj_path) if os.path.isfile(os.path.join(subj_path, sample)) and sample.endswith(args.file_ext)])
            samples_paths = [os.path.join(subj_path, sample) for sample in samples_file_name]

            embedds_subj = np.zeros((len(samples_paths),1,512), dtype=np.float32)
            for idx_sample, sample_path in enumerate(samples_paths):
                sample_name = os.path.basename(sample_path).split('.')[0]
                embedd_file_name = f'{sample_name}_embedding_r100_arcface.npy'
                embedd_file_path = os.path.join(output_subj_path, embedd_file_name)

                if not os.path.isfile(embedd_file_path):
                    print(f'    Computing and saving 2D embeddings - {idx_sample}/{len(samples_paths)}', end='\r')
                    img_tensor = load_sample(sample_path)
                    img_tensor = img_tensor.cuda()[None]
                    embedd_normalized = F.normalize(arcface(img_tensor))
                    embedd_normalized = embedd_normalized.cpu().detach().numpy()
                    embedds_subj[idx_sample] = embedd_normalized
                else:
                    print(f'    Loading 2D embeddings - {idx_sample}/{len(samples_paths)}', end='\r')
                    embedd_normalized = np.load(embedd_file_path)
                    embedds_subj[idx_sample] = embedd_normalized

                if not args.dont_replace_existing_files:
                    np.save(embedd_file_path, embedd_normalized)
                # print('\nembedd.shape:', embedd.shape)
                # sys.exit(0)
            print('')

            mean_embedd_subj = embedds_subj.mean(axis=0)
            # std_embedd_subj = embedds_subj.std(axis=0)
            mean_embedd_file_name = f'{subj}_mean_embedding_r100_arcface.npy'
            mean_embedd_file_path = os.path.join(output_subj_path, mean_embedd_file_name)
            print(f'    Saving mean embedding: \'{mean_embedd_file_path}\'')
            np.save(mean_embedd_file_path, mean_embedd_subj)
            # sys.exit(0)

            subj_elapsed_time = (time.time() - subj_start_time)
            print('    subj_elapsed_time: %.2f sec' % (subj_elapsed_time))
            print('---------------------')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/3D_reconstruction_MICA/real/1_CASIA-WebFace/imgs_crops_112x112_ONLY-SAMPLED/arcface')
    
    # parser.add_argument('--str_begin', default='', type=str, help='Substring to find and start processing')
    # parser.add_argument('--str_end', default='', type=str, help='Substring to find and stop processing')
    # parser.add_argument('--str_pattern', default='', type=str, help='Substring to find and stop processing')

    parser.add_argument('--divs', default=1, type=int, help='How many parts to divide paths list (useful to paralelize process)')
    parser.add_argument('--part', default=0, type=int, help='Specific part to process (works only if -div > 1)')

    parser.add_argument('--file_ext', default='.jpg', type=str, help='.npy, .jpg')

    parser.add_argument('--dont_replace_existing_files', action='store_true', help='')

    args = parser.parse_args()

    main(args)
