import argparse
import os, sys
import random
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from loguru import logger
from pytorch3d.io import save_ply, save_obj
from skimage.io import imread
from tqdm import tqdm

from configs.config import get_cfg_defaults
from datasets.creation.util import get_arcface_input, get_center
from utils import util


def parse_args(argv):
    parser = argparse.ArgumentParser(description='visualize_cropped_faces.py')

    parser.add_argument('-i1', default='/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/arcface/lfw_cfp_agedb/cfp/imgs/0.npy', type=str, help='Input file (.jpg, .png, .npy)')
    # parser.add_argument('-i2', default='/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/arcface/lfw_cfp_agedb/cfp/imgs/0.npy', type=str, help='Input file (.jpg, .png, .npy)')
    parser.add_argument('-i2', default='/datasets2/pbqv20/cfp_bkp/imgs/0.npy', type=str, help='Input file (.jpg, .png, .npy)')
    parser.add_argument('-l1', default=1, type=int, help='Backward level of dirs to search files')
    parser.add_argument('-l2', default=1, type=int, help='Backward level of dirs to search files')
    # parser.add_argument('-p', action='store_true', help='Print file metadata')
    # parser.add_argument('-un', action='store_true', help='Unnormalize image (for binary files)')
    parser.add_argument('-metric', default='MAE', type=str, help='Error metric')
    parser.add_argument('-dataset', default='CFP-FP', type=str, help='Error metric')
    

    args = parser.parse_args()
    # print('args:', args)
    # sys.exit(0)
    return args


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


def load_all_neighboor_images(img_path, level=1):
    splited_path = os.path.splitext(img_path)
    img_file = img_path.split('/')[-1]
    img_ext = splited_path[1]
    path_dir_img = os.path.dirname(img_path)
    # print('img_file:', img_file, '    img_ext:', img_ext, '    path_dir_img:', path_dir_img)
    sub_dirs = path_dir_img.split('/')
    path_to_search = '/'.join(sub_dirs[:len(sub_dirs)-(level-1)]) + '/*'*level + img_ext
    print('path_to_search:', path_to_search)
    neighboor_paths = sorted(glob(path_to_search))
    # print('neighboor_paths', neighboor_paths)
    neighboor_file_names = [path.split('/')[-1] for path in neighboor_paths]
    return neighboor_paths, neighboor_file_names


def check_files_names(file_names1=[], file_names2=[]):
    assert len(file_names1) == len(file_names2)
    for i in range(len(file_names1)):
        if file_names1[i] != file_names2[i]:
            print('Error, file names are different!')
            print(f'i:{i} - file_name1: {file_names1[i]}')
            print(f'i:{i} - file_name2: {file_names2[i]}')
            print()
            sys.exit(0)


def load_img(path_img):
    if path_img.endswith('.npy'):
        img = np.load(path_img)
        img = ((img + 1.0) * 127.5).astype(np.uint8)
    if path_img.endswith('.jpg') or path_img.endswith('.png'):
        img = cv2.imread(path_img)

    if img.shape[0] == 3 and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
        # print('Transposing axis...')
        img = img.transpose(1, 2, 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def mean_abs_error(img1, img2):
    error = np.mean(np.absolute(img1.astype(float) - img2.astype(float)))
    return error


def compute_error_between_imgs(paths1=[], paths2=[], metric='MAE'):
    errors = np.zeros((len(paths1)))
    for i, (path1, path2) in enumerate(zip(paths1, paths2)):
        print(f'{i}/{len(paths1)-1}', end='\r', flush=True)
        img1 = load_img(path1)
        img2 = load_img(path2)
        
        if metric == 'MAE':
            errors[i] = mean_abs_error(img1, img2)
        else:
            print('Error, metric not implemented yet:', metric, '\n')
            sys.exit(1)
    
    return errors


def plot_errors_curve(data, title, y_label, save_path=None, save_to_disk=False, show_on_screen=False):
    fig, ax = plt.subplots()

    ax.plot(data)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel('Image index')
    ax.set_ylim(0, 255)

    if save_to_disk and save_path:
        plt.savefig(save_path)
    if show_on_screen:
        plt.show()
    plt.close()


def main(args):
    
    print('args.i1:', args.i1)

    print('Searching neighboors...')
    neighboor_paths1, neighboor_file_names1 = load_all_neighboor_images(args.i1, level=args.l1)
    neighboor_paths2, neighboor_file_names2 = load_all_neighboor_images(args.i2, level=args.l2)

    check_files_names(neighboor_file_names1, neighboor_file_names2)

    # for i in range(len(neighboor_paths1)):
    #     print(i)
    #     print(f'path1: {neighboor_paths1[i]} - filename1: {neighboor_file_names1[i]}')
    #     print(f'path2: {neighboor_paths2[i]} - filename2: {neighboor_file_names2[i]}')
    #     print('---------')
    
    print('Computing error between images...')
    errors = compute_error_between_imgs(neighboor_paths1, neighboor_paths2, args.metric)
    # print('errors:', errors)

    title = 'Error between corresponding images  -  dataset: ' + args.dataset
    path_figure = './errors_between_imgs_dataset=' + str(args.dataset) + '_metric=' + str(args.metric) + '.png'
    print('Plotting figure with errors:', path_figure)
    plot_errors_curve(errors, title, args.metric, path_figure, save_to_disk=True, show_on_screen=False)

    print('\nFinished')





if __name__ == '__main__':
    args = parse_args(sys.argv)

    main(args)
