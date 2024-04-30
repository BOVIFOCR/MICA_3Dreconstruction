import os
import sys

import argparse
import random
import socket
import time
import glob
import pickle

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
    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as file:
            dists = pickle.load(file)
    else:
        raise ValueError(f"Unsupported file format \'{file_path}\'")
    return dists
    

def flat_array_remove_invalid_values(array, invalid_value=-1):
    if isinstance(array, dict):
        dict_data = [array[key] for key in array.keys()]
        array = np.array(dict_data)

    flat_array = array.flatten()
    valid_values = flat_array[flat_array != invalid_value]
    return valid_values


def compute_metrics_distances_subject(dist_data):
    metrics = {}
    metrics['all_distances'] = dist_data
    metrics['mean'] = np.mean(dist_data)
    metrics['std'] = np.std(dist_data)
    return metrics


def merge_metrics_dists(metrics_dist_subj):
    num_all_distances = 0
    for i, key in enumerate(metrics_dist_subj.keys()):
        num_all_distances += metrics_dist_subj[key]['all_distances'].shape[0]

    idx_begin_all_dist, idx_end_all_dist = 0, 0
    all_distances = np.zeros((num_all_distances,), dtype=np.float32)
    means = np.zeros((len(metrics_dist_subj),), dtype=np.float32)
    stds = np.zeros((len(metrics_dist_subj),), dtype=np.float32)
    for i, key in enumerate(metrics_dist_subj.keys()):
        idx_end_all_dist = idx_begin_all_dist + metrics_dist_subj[key]['all_distances'].shape[0]
        all_distances[idx_begin_all_dist:idx_end_all_dist] = metrics_dist_subj[key]['all_distances']
        idx_begin_all_dist = idx_end_all_dist
        
        means[i] = metrics_dist_subj[key]['mean']
        stds[i] = metrics_dist_subj[key]['std']
    return all_distances, means, stds


def save_histograms(all_distances, means, stds, filename, title):
    hist_all_dists, bin_all_dists_edges = np.histogram(all_distances, bins=20, density=True)
    bin_width = bin_all_dists_edges[1] - bin_all_dists_edges[0]
    plt.bar(bin_all_dists_edges[:-1], hist_all_dists/np.sum(hist_all_dists), width=bin_width, edgecolor='black', alpha=0.7, label='All dists')

    hist_means, bin_means_edges = np.histogram(means, bins=20, density=True)
    bin_width = bin_means_edges[1] - bin_means_edges[0]
    plt.bar(bin_means_edges[:-1], hist_means/np.sum(hist_means), width=bin_width, edgecolor='black', alpha=0.7, label='Means of dists')

    # hist_stds, bin_stds_edges = np.histogram(stds, bins=20, density=True)
    # bin_width = bin_stds_edges[1] - bin_stds_edges[0]
    # plt.bar(bin_stds_edges[:-1], hist_stds/np.sum(hist_stds), width=bin_width, edgecolor='black', alpha=0.7, label='Stds')


    # Plot histograms
    # plt.hist(means, bins=20, density=False, stacked=True, alpha=0.5, label='Means')
    # plt.hist(stds, bins=20, density=True, stacked=True, alpha=0.5, label='Stds')

    # Add title, labels, and legend
    plt.title(title)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend()

    plt.xlim([0, 1])
    # plt.xlim([0, 20])
    plt.ylim([0, 0.5])

    # Save the plot as PNG
    plt.savefig(filename)

    # Show the plot (optional)
    plt.show()


def find_files_by_extension(folder_path, extension, ignore_file_with=''):
    matching_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Check if the file ends with the specified extension
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                if ignore_file_with == '' or not ignore_file_with in file_path:
                    matching_files.append(file_path)
    return sorted(matching_files)


def main(args):
    img_path = args.img_path.rstrip('/')
    dist_path = args.dist_path.rstrip('/')
    print('img_path:', img_path)
    print('dist_path:', dist_path)

    output_kept_samples_dir = os.path.basename(img_path) + f'_outliersRemoved_keptSamples_metric={args.metric}_thresh={args.thresh}'
    output_kept_samples_path = os.path.join(os.path.dirname(img_path), output_kept_samples_dir)
    print('output_kept_samples_path:', output_kept_samples_path)
    os.makedirs(output_kept_samples_path, exist_ok=True)

    output_discard_samples_dir = os.path.basename(img_path) + f'_outliersRemoved_discardSamples_metric={args.metric}_thresh={args.thresh}'
    output_discard_samples_path = os.path.join(os.path.dirname(img_path), output_discard_samples_dir)
    print('output_discard_samples_path:', output_discard_samples_path)
    os.makedirs(output_discard_samples_path, exist_ok=True)

    print('---')
    print('Searching subject subfolders...')
    # subjects_paths = sorted([os.path.join(dist_path,subj) for subj in os.listdir(dist_path) if os.path.isdir(os.path.join(dataset_path, subj))])
    subjects_names = sorted([subj for subj in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, subj))])
    # print('subjects_names:', subjects_names)
    print(f'Found {len(subjects_names)} subjects!')
    # sys.exit(0)

    print('Filtering samples\n')
    for idx_subj, subj_name in enumerate(subjects_names):
        subj_start_time = time.time()

        # print(f'{idx_subj}/{len(subjects_names)} - Subject \'{subj_name}\'')
        subj_dist_path = os.path.join(dist_path, subj_name)
        file_dist_pattern = os.path.join(subj_dist_path, '*' + args.dist_ext)
        file_dist_path = glob.glob(file_dist_pattern)
        assert len(file_dist_path) > 0, f'Error, no file found with pattern \'{file_dist_pattern}\''
        file_dist_path = file_dist_path[0]
        # print('file_dist_path:', file_dist_path)
        # sys.exit(0)
        dists_metric = load_distances(file_dist_path)
        # print('dists_metric:', dists_metric)
        # sys.exit(0)

        for idx_sample_dist, key_sample_dist in enumerate(dists_metric.keys()):
            print(f'Subject {idx_subj}/{len(subjects_names)} - Sample {idx_sample_dist}/{len(dists_metric.keys())}')
            print('    key_sample_dist:', key_sample_dist)
            sample_dist = dists_metric[key_sample_dist]
            sample_name = key_sample_dist.split('/')[-1].split('_')[0]
            subj_img_path = os.path.join(img_path, subj_name)
            sample_img_pattern = os.path.join(subj_img_path, f"{sample_name}.{args.img_ext.lstrip('.')}")
            sample_img_path = glob.glob(sample_img_pattern)
            assert len(sample_img_path) > 0, f'Error, no file found with pattern \'{sample_img_pattern}\''
            sample_img_path = sample_img_path[0]
            print('    sample_img_path:', sample_img_path)
            # sys.exit(0)

            if sample_dist <= args.thresh:
                dst_dir = output_kept_samples_path
            else:
                dst_dir = output_discard_samples_path
            dst_subj_path = os.path.join(dst_dir, subj_name)
            os.makedirs(dst_subj_path, exist_ok=True)
            dst_sample_path = os.path.join(dst_subj_path, os.path.basename(sample_img_path))
            print('    symb link:', dst_sample_path)
            if not os.path.isfile(dst_sample_path):
                os.symlink(sample_img_path, dst_sample_path)

        print('-----')
        # sys.exit(0)

    print('\nFinished!')

        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path', type=str, default='/datasets2/2nd_frcsyn_cvpr2024/datasets/synthetic/dcface/dcface_0.5m_oversample_xid/imgs')
    parser.add_argument('--dist-path', type=str, default='/datasets2/2nd_frcsyn_cvpr2024/datasets/3D_reconstruction_MICA/synthetic/dcface/dcface_0.5m_oversample_xid/imgs/distances_cosine_2d')
    
    parser.add_argument('--img-ext', default='.png', type=str, help='.png')
    parser.add_argument('--dist-ext', default='.pkl', type=str, help='.pkl, .npy')
    
    parser.add_argument('--metric', default='cosine_2d', type=str, help='Options: chamfer, cosine_3dmm, cosine_2d')
    # parser.add_argument('--dataset_name', default='CASIA-WebFace', type=str, help='')
    parser.add_argument('--thresh', default=0.6, type=float, help='Minimum distance to mean embedding to be kept')

    args = parser.parse_args()

    main(args)
