import os
import sys

import argparse
import random
import socket
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pytorch3d.io import load_obj
from pytorch3d.loss import chamfer_distance
from mpl_toolkits.mplot3d import Axes3D

import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.io import load_obj, load_ply
from pytorch3d.loss import chamfer_distance


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
    else:
        raise ValueError("Unsupported file format. Only .obj and .ply files are supported.")
    return vertices
    

def compute_chamfer_distance(points1, points2):
    chamfer_dist = chamfer_distance(points1.unsqueeze(0), points2.unsqueeze(0))
    return chamfer_dist[0]


def compute_cosine_distance(array1, array2, normalize=True):
    if array1.shape[0] == 1:
        array1 = array1[0]
    if array2.shape[0] == 1:
        array2 = array2[0]
    
    if normalize == True:
        array1 = torch.nn.functional.normalize(array1, dim=0)
        array2 = torch.nn.functional.normalize(array2, dim=0)
    cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)(array1, array2)
    cos_dist = 1.0 - cos_sim
    return cos_dist


def compute_euclidean_distance(array1, array2, normalize=True):
    # print('array1.shape:', array1.shape)
    if normalize == True:
        array1 = torch.nn.functional.normalize(array1, dim=0)
        array2 = torch.nn.functional.normalize(array2, dim=0)
    eucl_dist = torch.norm(array1 - array2)
    return eucl_dist


def find_files_by_extension(folder_path, extension):
    matching_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Check if the file ends with the specified extension
            if file.endswith(extension):
                # If it does, add the full path to the list
                matching_files.append(os.path.join(root, file))
    return sorted(matching_files)



def main(args):
    assert args.part < args.divs, f'Error, args.part ({args.part}) >= args.divs ({args.divs}), but should be args.part ({args.part}) < args.divs ({args.divs})'

    dataset_path = args.input_path.rstrip('/')
    output_path = os.path.join(os.path.dirname(dataset_path), 'distances_'+args.metric)
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

    print('Computing distances...\n')
    for idx_subj, subj_path in enumerate(subjects_paths):
        if idx_subj >= idx_subj_begin and idx_subj < idx_subj_end:
            subj_start_time = time.time()

            subj = os.path.basename(subj_path)
            output_subj_path = os.path.join(output_path, subj)
            os.makedirs(output_subj_path, exist_ok=True)

            distances_file_name = 'distances_'+args.metric+'.npy'
            output_distances_path = os.path.join(output_subj_path, distances_file_name)
            if args.dont_replace_existing_files:
                if os.path.isfile(output_distances_path):
                    print(f'Skipping subject {idx_subj-idx_subj_begin}/{num_subjs_part} - \'{subj}\', distances file already exists: \'{output_distances_path}\'')
                    continue
            
            print(f'{idx_subj}/{len(subjects_paths)} - Searching subject samples in \'{subj_path}\'')
            samples_paths = find_files_by_extension(subj_path, args.file_ext)

            loaded_samples = [None] * len(samples_paths)
            for idx_sf, sample_path in enumerate(samples_paths):
                print(f'Loading samples - {idx_sf}/{len(samples_paths)}...', end='\r')
                data = load_sample(sample_path)
                loaded_samples[idx_sf] = data
            print('')
            # print('loaded_samples:', loaded_samples)
            # sys.exit(0)

            dist_matrix = -np.ones((len(loaded_samples),len(loaded_samples)), dtype=np.float32)
            for i in range(len(loaded_samples)):
                for j in range(i+1, len(loaded_samples)):
                    print(f'    Computing intra-class \'{args.metric}\' distances - i: {i}/{len(loaded_samples)}  j: {j}/{len(loaded_samples)}', end='\r')
                    sample1 = loaded_samples[i]
                    sample2 = loaded_samples[j]

                    if args.metric == 'chamfer':
                        dist = compute_chamfer_distance(sample1, sample2)
                    elif args.metric == 'cosine_3dmm' or args.metric == 'cosine_2d':
                        dist = compute_cosine_distance(sample1, sample2)
                    elif args.metric == 'euclidean_3dmm':
                        dist = compute_euclidean_distance(sample1, sample2, normalize=False)

                    # chamfer_distances.append(chamfer_dist)
                    # print('dist:', dist)
                    dist_matrix[i,j] = dist
            print('')

            print(f'    Saving distances file: \'{output_distances_path}\'')
            np.save(output_distances_path, dist_matrix)

            subj_elapsed_time = (time.time() - subj_start_time)
            print('    subj_elapsed_time: %.2f sec' % (subj_elapsed_time))
            print('---------------------')

    print('\nFinished!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/3D_reconstruction_MICA/real/1_CASIA-WebFace/imgs_crops_112x112_ONLY-SAMPLED/output')
    
    # parser.add_argument('--str_begin', default='', type=str, help='Substring to find and start processing')
    # parser.add_argument('--str_end', default='', type=str, help='Substring to find and stop processing')
    # parser.add_argument('--str_pattern', default='', type=str, help='Substring to find and stop processing')

    parser.add_argument('--divs', default=1, type=int, help='How many parts to divide paths list (useful to paralelize process)')
    parser.add_argument('--part', default=0, type=int, help='Specific part to process (works only if -div > 1)')

    parser.add_argument('--metric', default='euclidean_3dmm', type=str, help='Options: chamfer, cosine_3dmm, euclidean_3dmm, cosine_2d')
    parser.add_argument('--file_ext', default='.ply', type=str, help='.ply, .obj, .npy')

    parser.add_argument('--dont_replace_existing_files', action='store_true', help='')

    args = parser.parse_args()

    main(args)
