import os
import sys
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt

import mxnet as mx
from mxnet import ndarray as nd
from mxnet.io import ImageRecordIter
from mxnet.base import MXNetError


def load_rec(rec_path, idx_path, lst_path, batch_size, img_size):
    train_data = ImageRecordIter(
        # path_imgrec = os.path.join(rec_path, 'train.rec'),
        # path_imgidx = os.path.join(rec_path, 'train.idx'),
        path_imgrec = rec_path,
        path_imgidx = idx_path,
        # path_imglist = lst_path,
        # data_shape  = (3, 224, 224),
        data_shape  = (3, img_size[0], img_size[1]),
        label_width = 2,
        # batch_size  = 32,
        batch_size  = batch_size,
        # shuffle     = True,
        preprocess_threads = 1
    )
    return train_data


def load_bin(path, image_size):
    '''
    Loads validation datasets for verification
    '''
    # bins, issame_list = pickle.load(open(path, 'rb'))                       # original
    # bins, issame_list = pickle.load(open(path, 'rb'), encoding='latin1')    # Bernardo
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')       # Bernardo
    
    data_list = []
    for flip in [0,1]:
        data = nd.empty((len(issame_list)*2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for i in range(len(issame_list)*2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin)
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0,1]:
            if flip==1:
                img = mx.ndarray.flip(data=img, axis=2)
                # print('img.shape:', img.shape)
                # cv2.imshow('img', cv2.cvtColor(np.swapaxes(np.swapaxes(img.asnumpy(),0,1),1,2), cv2.COLOR_RGB2BGR))
                # cv2.waitKey(0)
            data_list[flip][i][:] = img
        if i%1000==0:
            print('loading bin', i)
    print(data_list[0].shape)
    return (data_list, issame_list)



def extract_images_from_binary_file(train_data, lst_data_path, output_path, batch_size):
    lst_file = open(lst_data_path, 'r')

    train_data.reset()
    i = 0
    while True:
        try:
            batch = train_data.next()
            batch_img = batch.data[0]
            batch_label = batch.label[0]
            for j in range(batch_img.shape[0]):
                line = lst_file.readline().strip('\n')
                print('line:', line)
                print('i:', i, '    batch_img.shape[0]:', batch_img.shape[0], '    j:', j)
                subj_id = line.split('\t')[-1]
                orig_img_path = line.split('\t')[1]
                dest_dir = orig_img_path.split('/')[-2]
                dest_path_dir = output_path + '/' + dest_dir

                if not os.path.isdir(dest_path_dir):
                    os.makedirs(dest_path_dir)
                dest_img_path = dest_path_dir + '/' + orig_img_path.split('/')[-1].replace('.jpg', '.png')

                img_index = i*batch_size+j
                print('Saving img_index:', img_index, ' - subj_id:', subj_id, ' - ', dest_img_path)
                img_data = cv2.cvtColor(batch_img[j].asnumpy().astype(np.uint8).transpose((1,2,0)), cv2.COLOR_RGB2BGR)
                cv2.imwrite(dest_img_path, img_data)

                # if 'm.03png_b' in dest_img_path:
                #     input('PAUSED')

                print('-----------------------')
                # input('PAUSED')

            i += 1

        except MXNetError as e:
            break




if __name__ == '__main__':
    # data_file_path = '/media/biesseck/DATA/BernardoBiesseck/BOVIFOCR_project/datasets/MS-Celeb-1M/ms1m-retinaface-t1/lfw.bin'
    # # data_file_path = '/media/biesseck/DATA/BernardoBiesseck/BOVIFOCR_project/datasets/MS-Celeb-1M/ms1m-retinaface-t1/agedb_30.bin'
    # # data_file_path = '/media/biesseck/DATA/BernardoBiesseck/BOVIFOCR_project/datasets/MS-Celeb-1M/ms1m-retinaface-t1/cfp_fp.bin'
    
    # img_size = [112, 112]
    
    # load_bin(data_file_path, img_size)




    rec_data_path = '/home/bjgbiesseck/datasets/MS-Celeb-1M/ms1m-retinaface-t1/train.rec'
    idx_data_path = '/home/bjgbiesseck/datasets/MS-Celeb-1M/ms1m-retinaface-t1/train.idx'
    lst_data_path = '/home/bjgbiesseck/datasets/MS-Celeb-1M/ms1m-retinaface-t1/train.lst'

    output_path = '/home/bjgbiesseck/datasets/MS-Celeb-1M/ms1m-retinaface-t1/images'

    # img_size = [28, 28]
    # img_size = [56, 56]
    img_size = [112, 112]
    # img_size = [224, 224]
    # img_size = [512, 512]
    # img_size = [1024, 1024]
    # img_size = [2048, 2048]

    batch_size = 32

    train_data = load_rec(rec_data_path, idx_data_path, lst_data_path, batch_size, img_size)
    extract_images_from_binary_file(train_data, lst_data_path, output_path, batch_size)
    
    print('\nFinished!!!')
