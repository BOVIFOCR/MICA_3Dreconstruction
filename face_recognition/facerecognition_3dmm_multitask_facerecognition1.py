import os
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

# # from jobs import train
# from jobs import train_multitask_facerecognition1

from yacs.config import CfgNode as CN

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from micalib.tester_multitask_facerecognition1 import TesterMultitaskFacerecognition1     # Bernardo
from utils import util

from face_classifier_3dmm import FaceClassifier1_MLP

from torchvision import transforms
from dataloaders.lfw import LFW_3DMM
from dataloaders.mlfw import MLFW_3DMM

# BERNARDO
import argparse
import numpy as np
import random
import socket
host_name = socket.gethostname()

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file',     type=str, help='YML file path', required=True)
    parser.add_argument('--experiment',   type=str, help='Model folder path', required=True)
    parser.add_argument('--checkpoint',   type=str, help='TAR weights model file', default='model.tar', required=True)

    parser.add_argument('--train_dataset_name', type=str, help='', default='', required=True)
    parser.add_argument('--train_dataset_path', type=str, help='', default='', required=True)
    parser.add_argument('--test_dataset_name', type=str, help='', default='', required=True)
    parser.add_argument('--test_dataset_path', type=str, help='', default='', required=True)
    
    parser.add_argument('--file_ext',     type=str, help='File extension of embbeding face', default='', required=True)

    args = parser.parse_args()
    return args


def load_mica_multitask_facerecognition1():
    cfg = CN()
    cfg.model = CN()
    cfg.model.name = 'micamultitaskfacerecognition1'
    cfg.model.testing = True
    cfg.train = CN()
    cfg.train.use_mask = False
    cfg.mask_weights = CN()
    rank = 1
    # mica = util.find_model_using_name(model_dir='micalib.models', model_name=cfg.model.name)(cfg, rank)  # original
    mica = util.find_model_using_name(model_dir='micalib.models', model_name=cfg.model.name)(cfg, rank)    # Bernardo
    # tester = Tester(nfc_model=mica, config=cfg, device=rank)                          # original
    tester = TesterMultitaskFacerecognition1(nfc_model=mica, config=cfg, device=rank)   # Bernardo


def make_face_recognizer1_3dmm(num_classes=None):
    model_cfg = None
    device = 'cuda:0'
    # device_id = '0'
    faceClassifier = FaceClassifier1_MLP(num_classes=num_classes, model_cfg=model_cfg, device=device).to(device)
    return faceClassifier


def get_dataset_loader(dataset_name='', dataset_path=''):
    if dataset_name == 'LFW':
        dataset = LFW_3DMM(root_dir='/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/lfw', file_ext='identity.npy')

    if dataset_name == 'MLFW':
        dataset = MLFW_3DMM(root_dir='/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/MLFW', file_ext='identity.npy')

    elif dataset_name == 'MS1MV2':
        pass

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    return trainloader, dataset.num_classes, dataset.num_samples


def train_model(model=None, trainloader=None):
    # Define the loss function and optimizer
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Run the training loop
    for epoch in range(0, 10): # 5 epochs at maximum
        print(f'Starting epoch {epoch+1}')
        current_loss = 0.0
        
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            optimizer.zero_grad()
            logits = model(inputs)
            
            loss = loss_function(logits, targets)
            loss.backward()
            optimizer.step()
            
            current_loss += loss.item()
            if i % 100 == 99:
                print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 100))
                current_loss = 0.0

    # Process is complete.
    print('Training process has finished.')


def run_training(args):
    # load test dataset
    print('Getting dataset loader...')
    trainloader, num_classes, num_samples = get_dataset_loader(dataset_name=args.train_dataset_name, dataset_path=args.train_dataset_path)
    print('num_classes:', num_classes, '    num_samples:', num_samples)

    # load trained model
    print('Loading model...')
    # load_mica_multitask_facerecognition1()
    faceRecognizer_3dmm = make_face_recognizer1_3dmm(num_classes)

    # train model
    print('Training model...')
    train_model(model=faceRecognizer_3dmm, trainloader=trainloader)

    # do verification

    # save outputs


if __name__ == '__main__':

    sys.argv.append('--cfg_file')
    sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/11_mica_duo_MULTITASK-VALIDATION-WORKING_train=FRGC,LYHM,FLORENCE,FACEWAREHOUSE_eval=Stirling_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_lamb1=0.5_lamb2=1.0.yml')

    sys.argv.append('--experiment')
    sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/output/11_mica_duo_MULTITASK-VALIDATION-WORKING_train=FRGC,LYHM,FLORENCE,FACEWAREHOUSE_eval=Stirling_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_lamb1=0.5_lamb2=1.0')

    sys.argv.append('--checkpoint')
    sys.argv.append('model.tar')

    sys.argv.append('--train_dataset_name')
    sys.argv.append('LFW')
    # sys.argv.append('MLFW')
    # sys.argv.append('MS1MV2_1000')

    sys.argv.append('--train_dataset_path')
    # sys.argv.append('LFW')
    sys.argv.append('MLFW')

    sys.argv.append('--test_dataset_name')
    sys.argv.append('LFW')
    # sys.argv.append('MLFW')

    sys.argv.append('--test_dataset_path')
    # sys.argv.append('LFW')
    sys.argv.append('MLFW')

    sys.argv.append('--file_ext')
    sys.argv.append('.npy')


    # parse args
    args = parse_args(args=sys.argv)
    # print('args:', args)


    num_gpus = torch.cuda.device_count()

    torch.multiprocessing.set_start_method('spawn', force=True)
    run_training(args)

    # mp.spawn(run_training, args=(num_gpus, args), nprocs=1, join=True)
