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
import numpy as np
from torch.utils.data import ConcatDataset
import torch

from datasets.base import BaseDataset
from datasets.base_multitask_facerecognition import BaseDataset_MultitaskFaceRecognition


# Bernardo
def get_imagelabel_from_imagename(imagename, labels_map):
    imagelabel = [None] * len(imagename)
    for i, name in enumerate(imagename):
        imagelabel[i] = labels_map[name]
    return np.array(imagelabel)    

# Bernardo
def get_onehotvector_from_imagelabel1(imagelabel, num_classes):
    one_hot_labels = np.eye(num_classes)[imagelabel]
    return one_hot_labels

# Bernardo
def get_onehotvector_from_imagelabel2(imagelabel, num_classes):
    # one_hot_labels = torch.nn.functional.one_hot(imagelabel, num_classes)
    one_hot_labels = torch.nn.functional.one_hot(torch.from_numpy(imagelabel), num_classes)
    return one_hot_labels

# Bernardo
def get_all_indexes_in_list(value, values_list):
    indexes = np.where(np.array(values_list) == value)[0]
    return indexes

# Bernardo
def get_labels_map(train_dataset, val_dataset):
    i = 0
    all_actors_name = []
    labels_map = {}

    # train datasets
    for j, dataset in enumerate(train_dataset.datasets):
        # print('get_labels_map - j:', j, '    train_dataset.dataset.__dir__():', dataset.__dir__())
        # print('get_labels_map - j:', j, '    train_dataset.dataset.face_dict.keys():', list(dataset.face_dict.keys()))
        actors_name = list(dataset.face_dict.keys())
        all_actors_name += actors_name
        for name in actors_name:
            labels_map[name] = i
            i += 1

    # print('get_labels_map - self.labels_map:', labels_map)
    return labels_map


# Bernardo
def build_train_multitask_facerecognition(config, device):
    data_list = []
    total_images = 0
    for dataset in config.training_data:
        dataset_name = dataset.upper()
        config.n_train = np.Inf
        if type(dataset) is list:
            dataset_name, n_train = dataset
            config.n_train = n_train

        # dataset = BaseDataset(name=dataset_name, config=config, device=device, isEval=False)
        dataset = BaseDataset_MultitaskFaceRecognition(name=dataset_name, config=config, device=device, isEval=False)    # Bernardo

        data_list.append(dataset)
        total_images += dataset.total_images

    return ConcatDataset(data_list), total_images


# Bernardo
def build_val_multitask_facerecognition(config, device):
    data_list = []
    total_images = 0
    for dataset in config.eval_data:
        dataset_name = dataset.upper()
        config.n_train = np.Inf
        if type(dataset) is list:
            dataset_name, n_train = dataset
            config.n_train = n_train

        # dataset = BaseDataset(name=dataset_name, config=config, device=device, isEval=True)
        dataset = BaseDataset_MultitaskFaceRecognition(name=dataset_name, config=config, device=device, isEval=True)    # Bernardo
        # dataset = BaseDataset_MultitaskFaceRecognition(name=dataset_name, config=config, device=device, isEval=False)    # Bernardo

        data_list.append(dataset)
        total_images += dataset.total_images

    return ConcatDataset(data_list), total_images




def build_train(config, device):
    data_list = []
    total_images = 0
    for dataset in config.training_data:
        dataset_name = dataset.upper()
        config.n_train = np.Inf
        if type(dataset) is list:
            dataset_name, n_train = dataset
            config.n_train = n_train

        dataset = BaseDataset(name=dataset_name, config=config, device=device, isEval=False)

        data_list.append(dataset)
        total_images += dataset.total_images

    return ConcatDataset(data_list), total_images


def build_val(config, device):
    data_list = []
    total_images = 0
    for dataset in config.eval_data:
        dataset_name = dataset.upper()
        config.n_train = np.Inf
        if type(dataset) is list:
            dataset_name, n_train = dataset
            config.n_train = n_train

        dataset = BaseDataset(name=dataset_name, config=config, device=device, isEval=True)
        data_list.append(dataset)
        total_images += dataset.total_images

    return ConcatDataset(data_list), total_images
