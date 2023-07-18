import argparse
import os, sys
import random
from glob import glob
from pathlib import Path

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

    parser.add_argument('-i', default='/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/arcface/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.npy', type=str, help='Input file (.jpg, .png, .npy)')
    parser.add_argument('-un', action='store_true', help='')
    parser.add_argument('-p', action='store_true', help='')
    parser.add_argument('-l', default=0, type=int, help='')
    
    
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


def show_image(img, title='img'):
    ENTER = 13
    ESC = 27

    cv2.imshow(title, img)
    while True:
        keyCode = cv2.waitKey(1)
        # print('keyCode:', keyCode)
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) == 0 or keyCode == ENTER or keyCode == ESC:
            break
    cv2.destroyAllWindows()


def load_img(path_img):
    if path_img.endswith('.npy'):
        img = np.load(path_img)
    if path_img.endswith('.jpg') or args.i.endswith('.png'):
        img = cv2.imread(path_img)
    return img


def load_show_one_img(args):
    print('Loading \'' + args.i + '\'')
    img = load_img(args.i)
    if args.i.endswith('.npy'):
        args.un = True

    print('Original img.shape:', img.shape)
    if img.shape[0] == 3 and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
        print('Transposing axis...')
        img = img.transpose(1, 2, 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print('Current img.shape: ', img.shape)

    if args.un == True:
        print('Unnormalizing image...')
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        
    if args.p == True:
        print('img:', img)
    
    print('Showing image...')
    show_image(img)
    # cv2.imshow('img', img)
    # cv2.waitKey()


def load_all_neighboor_images(img_path, level=1, limit=-1):
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
    return neighboor_paths


def show_image_neighboors(args, paths, idx_img):
    ENTER = 13
    ESC = 27
    LEFT = 81
    RIGHT = 83

    print_img_path = True

    while True:
        img_path = paths[idx_img]

        if print_img_path:
            print('----------')
            print('idx_img: ', idx_img)
            print('Loading \'' + img_path + '\'')
        
        img = load_img(img_path)
        if img_path.endswith('.npy'):
            args.un = True

        if print_img_path:
            print('Original img.shape:', img.shape)

        if img.shape[0] == 3 and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
            if print_img_path:
                print('Transposing axis...')
            img = img.transpose(1, 2, 0)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if print_img_path:
            print('Current img.shape: ', img.shape)

        if args.un == True:
            if print_img_path:
                print('Unnormalizing image...')
            img = ((img + 1.0) * 127.5).astype(np.uint8)

        if args.p == True:
            if print_img_path:
                print('img:', img)

        if print_img_path:
            print('Showing image...')
        title = 'img'
        cv2.imshow(title, img)

        keyCode = cv2.waitKey(1)
        # print('keyCode:', keyCode)
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) == 0 or keyCode == ENTER or keyCode == ESC:
            break
        
        # if print_img_path:
        #     print('----------')

        print_img_path = False
        if len(paths) > 0:
            if keyCode == LEFT:
                idx_img -= 1
                if idx_img < 0:
                    idx_img = len(paths) - 1
                print_img_path = True
            elif keyCode == RIGHT:
                idx_img += 1
                if idx_img == len(paths):
                    idx_img = 0
                print_img_path = True

    cv2.destroyAllWindows()


def load_show_img_neighboors(args, limit):
    print('Loading neighboor files...')
    neighboor_paths = load_all_neighboor_images(args.i, args.l, limit)
    print('Neighboors:', len(neighboor_paths))

    if len(neighboor_paths) > 0:
        idx_curr_img = neighboor_paths.index(args.i)
        show_image_neighboors(args, neighboor_paths, idx_curr_img)


def main(args):

    if args.l > 0:
        limit = -1  # load all neighboors
        load_show_img_neighboors(args, limit)

    else:
        load_show_one_img(args)

    print('\nFinished')

    



if __name__ == '__main__':
    args = parse_args(sys.argv)

    main(args)
