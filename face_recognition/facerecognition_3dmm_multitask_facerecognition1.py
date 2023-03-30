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
from face_classifier_3dmm import ArcFaceLoss
import losses

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
    # parser.add_argument('--test_dataset_path', type=str, help='', default='', required=True)
    
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

    # faceClassifier = FaceClassifier1_MLP(num_classes=num_classes, model_cfg=model_cfg, device=device).to(device)
    faceClassifier = ArcFaceLoss(num_classes=num_classes, margin=0.5, scale=32, model_cfg=model_cfg, device=device).to(device)

    return faceClassifier


def get_dataset_loader(dataset_name='', dataset_path=''):
    if dataset_name == 'LFW':
        dataset = LFW_3DMM(root_dir=dataset_path, file_ext='identity.npy')

    if dataset_name == 'MLFW':
        dataset = MLFW_3DMM(root_dir=dataset_path, file_ext='identity.npy')

    elif dataset_name == 'MS1MV2':
        pass

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)
    return trainloader, valloader, dataset.num_classes, dataset.num_samples


def train_model(model=None, trainloader=None, valloader=None):
    # Cross entropy loss
    loss_function = losses.cross_entropy_loss_logits_and_targets()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=2e-5)

    # # Arcface loss
    # loss_function = model.get_arcface_loss
    # # optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)   # LFW
    # optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-5)   # MLFW

    # Run the training loop
    for epoch in range(0, 30):
        print(f'Starting epoch {epoch+1}')
        
        # Iterate over the DataLoader for training data
        train_loss = 0.0
        with torch.enable_grad():
            for i, data in enumerate(trainloader, 0):
                inputs, targets = data
                optimizer.zero_grad()
                logits = model(inputs)

                loss = loss_function(logits, targets)
                loss.backward()
                optimizer.step()

                # PRINT GRADIENT
                print('model.layers:', model.layers)
                # print('model.layers[1].weight:', model.layers[1].weight)
                print('model.layers[1].weight.grad:', model.layers[1].weight.grad)
                print('torch.linalg.matrix_norm(model.layers[1].weight.grad):', torch.linalg.matrix_norm(model.layers[1].weight.grad))

                train_loss += loss.item()
                if i % 100 == 99:
                    print('Train loss (mini-batch) %5d: %.3f' % (i + 1, train_loss / 100))
                    train_loss = 0.0
        
        # Iterate over the DataLoader for validation data
        val_loss = 0.0
        with torch.no_grad():
            num_val_samples = 0
            for i, valdata in enumerate(valloader, 0):
                inputs, targets = valdata
                logits = model(inputs)
                loss = loss_function(logits, targets)
                
                val_loss += loss.item()
                num_val_samples += 1

                # if i % 100 == 99:
                #     print('Val loss (mini-batch) %5d: %.3f' % (i + 1, val_loss / 100))
                #     val_loss = 0.0

            print('Val loss: %.3f' % (val_loss / num_val_samples))
        print('-----------------------')

    # Process is complete.
    print('Training process has finished.')


def run_training(args):
    # load test dataset
    print('Getting dataset loader:', args.train_dataset_name)
    print('args.train_dataset_path:', args.train_dataset_path)
    # trainloader, num_classes, num_samples = get_dataset_loader(dataset_name=args.train_dataset_name, dataset_path=args.train_dataset_path)
    trainloader, valloader, num_classes, num_samples = get_dataset_loader(dataset_name=args.train_dataset_name, dataset_path=args.train_dataset_path)
    print('num_classes:', num_classes, '    num_samples:', num_samples)

    # load trained model
    print('\nLoading model...')
    # load_mica_multitask_facerecognition1()
    faceRecognizer_3dmm = make_face_recognizer1_3dmm(num_classes)

    # train model
    print('\nTraining model...')
    train_model(model=faceRecognizer_3dmm, trainloader=trainloader, valloader=valloader)

    # do verification

    # save outputs


if __name__ == '__main__':

    exp = ''   # original MICA
    # exp = '11_mica_duo_MULTITASK-VALIDATION-WORKING_train=FRGC,LYHM,FLORENCE,FACEWAREHOUSE_eval=Stirling_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_lamb1=0.5_lamb2=1.0'

    sys.argv.append('--cfg_file')
    sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/configs/' + exp + '.yml')

    sys.argv.append('--experiment')
    sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/output/' + exp)

    sys.argv.append('--checkpoint')
    sys.argv.append('model.tar')

    sys.argv.append('--train_dataset_name')
    # sys.argv.append('LFW')
    sys.argv.append('MLFW')
    # sys.argv.append('MS1MV2_1000')

    sys.argv.append('--train_dataset_path')
    # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/lfw')   # original MICA
    # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/MLFW')    # original MICA
    # sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output_12_mica_duo_MULTITASK-VALIDATION-WORKING_train=FRGC,LYHM,Stirling,FACEWAREHOUSE_eval=FLORENCE_pretrainedMICA=False_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_lamb1=0.5_lamb2=1.0/MLFW')
    sys.argv.append('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output_12_mica_duo_MULTITASK-VALIDATION-WORKING_train=FRGC,LYHM,Stirling,FACEWAREHOUSE_eval=FLORENCE_pretrainedMICA=True_pretrainedARCFACE=ms1mv3-r100_fr-feat=3dmm_fr-lr=1e-5_lamb1=0.5_lamb2=1.0/MLFW')

    sys.argv.append('--test_dataset_name')
    sys.argv.append('LFW')
    # sys.argv.append('MLFW')

    # sys.argv.append('--test_dataset_path')
    # # sys.argv.append('LFW')
    # sys.argv.append('MLFW')

    sys.argv.append('--file_ext')
    sys.argv.append('.npy')


    # parse args
    args = parse_args(args=sys.argv)
    # print('args:', args)


    num_gpus = torch.cuda.device_count()

    torch.multiprocessing.set_start_method('spawn', force=True)
    run_training(args)

    # mp.spawn(run_training, args=(num_gpus, args), nprocs=1, join=True)
