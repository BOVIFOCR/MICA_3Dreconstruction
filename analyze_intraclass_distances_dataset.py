import os
import sys

import argparse
import random
import socket
import time
import glob

import numpy as np
import matplotlib.pyplot as plt

# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# from pytorch3d.io import load_obj
# from pytorch3d.loss import chamfer_distance
# from mpl_toolkits.mplot3d import Axes3D

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from pytorch3d.io import load_obj, load_ply
# from pytorch3d.loss import chamfer_distance


def load_distances(file_path):
    if file_path.endswith('.npy'):
        dists = np.load(file_path)
    else:
        raise ValueError(f"Unsupported file format \'{file_path}\'")
    return dists
    

def flat_array_remove_invalid_values(array, invalid_value=-1):
    flat_array = array.flatten()
    valid_values = flat_array[flat_array != invalid_value]
    return valid_values


def compute_metrics_distances_subject(dist_data):
    metrics = {}
    metrics['mean'] = np.mean(dist_data)
    metrics['std'] = np.std(dist_data)
    return metrics


def merge_metrics_dists(metrics_dist_subj):
    means = np.zeros((len(metrics_dist_subj),), dtype=np.float32)
    stds = np.zeros((len(metrics_dist_subj),), dtype=np.float32)
    for i, key in enumerate(metrics_dist_subj.keys()):
        means[i] = metrics_dist_subj[key]['mean']
        stds[i] = metrics_dist_subj[key]['std']
    return means, stds 


def save_histograms(means, stds, filename, title):
    hist_means, bin_means_edges = np.histogram(means, bins=20, density=True)
    bin_width = bin_means_edges[1] - bin_means_edges[0]
    plt.bar(bin_means_edges[:-1], hist_means/np.sum(hist_means), width=bin_width, edgecolor='black', alpha=0.7, label='Means')

    hist_stds, bin_stds_edges = np.histogram(stds, bins=20, density=True)
    bin_width = bin_stds_edges[1] - bin_stds_edges[0]
    plt.bar(bin_stds_edges[:-1], hist_stds/np.sum(hist_stds), width=bin_width, edgecolor='black', alpha=0.7, label='Stds')
    

    # Plot histograms
    # plt.hist(means, bins=20, density=False, stacked=True, alpha=0.5, label='Means')
    # plt.hist(stds, bins=20, density=True, stacked=True, alpha=0.5, label='Stds')

    # Add title, labels, and legend
    plt.title(title)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend()

    # plt.xlim([0, 1])
    plt.xlim([0, 10])
    plt.ylim([0, 0.5])

    # Save the plot as PNG
    plt.savefig(filename)

    # Show the plot (optional)
    plt.show()


def main(args):
    dataset_path = args.input_path.rstrip('/')
    output_path = os.path.dirname(dataset_path)
    # os.makedirs(output_path, exist_ok=True)

    print('dataset_path:', dataset_path)
    print('Searching subject subfolders...')
    subjects_paths = sorted([os.path.join(dataset_path,subj) for subj in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, subj))])
    # print('subjects_paths:', subjects_paths)
    print(f'Found {len(subjects_paths)} subjects!')
    # sys.exit(0)

    metrics_dist_subj = {}
    print('Loading distances...\n')
    for idx_subj, subj_path in enumerate(subjects_paths):
        subj_start_time = time.time()
        
        subj_name = os.path.basename(subj_path)
        print(f'{idx_subj}/{len(subjects_paths)} - Loading subject {subj_name}', end='\r')

        file_pattern = os.path.join(subj_path, '*' + args.file_ext)
        dist_file_path = glob.glob(file_pattern)
        if len(dist_file_path) > 0:
            # assert len(dist_file_path) > 0, f'Error, file not found: \'{file_pattern}\''
            dist_file_path = dist_file_path[0]
            dist_data = load_distances(dist_file_path)
            # print('dist_data.shape:', dist_data.shape)
            # sys.exit(0)

            dist_data = flat_array_remove_invalid_values(dist_data, invalid_value=-1)
            # print('dist_data.shape:', dist_data.shape)
            # sys.exit(0)

            metrics_dist_subj[subj_name] = compute_metrics_distances_subject(dist_data)
            # print('metrics_dist_subj:', metrics_dist_subj)
            # sys.exit(0)
    print('')
    
    print('Merging metrics...')
    all_means_dist, all_stds_dist = merge_metrics_dists(metrics_dist_subj)
    # print('all_means_dist:', all_means_dist)
    # print('all_means_dist.shape:', all_means_dist.shape)

    title = f'dataset \'{args.dataset_name}\' - {len(metrics_dist_subj)} subjects - {args.metric}'
    chart_file_name = 'histograms_distances_' + args.metric + '.png'
    chart_file_path = os.path.join(output_path, chart_file_name)
    print(f'Saving histograms: \'{chart_file_path}\'')
    save_histograms(all_means_dist, all_stds_dist, chart_file_path, title)

    print('\nFinished!')

        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default='/datasets2/1st_frcsyn_wacv2024/datasets/3D_reconstruction_MICA/real/1_CASIA-WebFace/imgs_crops_112x112/distances_cosine_3dmm')
    
    parser.add_argument('--metric', default='cosine_3dmm', type=str, help='Options: chamfer, cosine_3dmm, cosine_2d')
    parser.add_argument('--file_ext', default='.npy', type=str, help='.npy')
    parser.add_argument('--dataset_name', default='CASIA-WebFace', type=str, help='')

    args = parser.parse_args()

    main(args)
