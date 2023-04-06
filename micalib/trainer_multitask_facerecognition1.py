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


import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
from configs.config import cfg
from utils import util

# from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("./micalib")
# from validator import Validator                                                      # original
from validator_multitask_facerecognition1 import ValidatorMultitaskFacerecognition1    # Bernardo

torch.autograd.set_detect_anomaly(True)


def print_info(rank):
    props = torch.cuda.get_device_properties(rank)

    logger.info(f'[INFO]            {torch.cuda.get_device_name(rank)}')
    logger.info(f'[INFO] Rank:      {str(rank)}')
    logger.info(f'[INFO] Memory:    {round(props.total_memory / 1024 ** 3, 1)} GB')
    logger.info(f'[INFO] Allocated: {round(torch.cuda.memory_allocated(rank) / 1024 ** 3, 1)} GB')
    logger.info(f'[INFO] Cached:    {round(torch.cuda.memory_reserved(rank) / 1024 ** 3, 1)} GB')


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TrainerMultitaskFacerecognition1(object):
    def __init__(self, nfc_model, config=None, device=None):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config

        logger.add(os.path.join(self.cfg.output_dir, self.cfg.train.log_dir, 'train_multitask_facerecognition1.log'))

        self.device = device
        self.batch_size = self.cfg.dataset.batch_size
        self.K = self.cfg.dataset.K

        # deca model
        self.nfc = nfc_model.to(self.device)

        # self.validator = Validator(self)                           # original
        self.validator = ValidatorMultitaskFacerecognition1(self)    # Bernardo
        self.configure_optimizers()
        self.load_checkpoint()

        # Bernardo
        self.labels_map = {}
        self.early_stop = False
        self.early_stop_tolerance = self.cfg.train.early_stop_tolerance   # std/avg (Bernardo)
        self.early_stop_patience = self.cfg.train.early_stop_patience     # Bernardo
        
        # reset optimizer if loaded from pretrained model
        if self.cfg.train.reset_optimizer:
            self.configure_optimizers()  # reset optimizer
            logger.info(f"[TRAINER] Optimizer was reset")

        if self.cfg.train.write_summary and self.device == 0:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=os.path.join(self.cfg.output_dir, self.cfg.train.log_dir))

        print_info(device)

    def configure_optimizers(self):
        # original
        # self.opt = torch.optim.AdamW(
        #     lr=self.cfg.train.lr,
        #     weight_decay=self.cfg.train.weight_decay,
        #     params=self.nfc.parameters_to_optimize(),
        #     amsgrad=False)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=1, gamma=0.1, verbose=True)   # original

        # Bernardo
        self.opt = torch.optim.SGD(
            params=self.nfc.parameters_to_optimize(),
            lr=self.cfg.train.lr, momentum=0.9, weight_decay=self.cfg.train.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.opt, T_max=3e+5, eta_min=self.cfg.train.lr, last_epoch=-1, verbose=True)

    def load_checkpoint(self):
        self.epoch = 0
        self.global_step = 0
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.device}
        model_path = os.path.join(self.cfg.output_dir, 'model.tar')

        # if os.path.exists(self.cfg.pretrained_model_path):                                   # original
        if os.path.exists(self.cfg.pretrained_model_path) and self.cfg.model.use_pretrained:   # Bernardo
            model_path = self.cfg.pretrained_model_path
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location)

            # COMMENTED BY BERNARDO DUE THE ORIGINAL MODEL HASN'T FACE CLASSIFICATION LAYERS
            # if 'opt' in checkpoint:
            #     self.opt.load_state_dict(checkpoint['opt'])

            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            if 'epoch' in checkpoint:
                self.epoch = checkpoint['epoch']
            if 'global_step' in checkpoint:
                self.global_step = checkpoint['global_step']

            logger.info(f"[TRAINER] Resume training from {model_path}")
            logger.info(f"[TRAINER] Start from step {self.global_step}")
            logger.info(f"[TRAINER] Start from epoch {self.epoch}")
        else:
            logger.info('[TRAINER] Model path not found, start training from scratch')

    def save_checkpoint(self, filename):
        if self.device == 0:
            model_dict = self.nfc.model_dict()

            model_dict['opt'] = self.opt.state_dict()
            model_dict['scheduler'] = self.scheduler.state_dict()
            model_dict['validator'] = self.validator.state_dict()
            model_dict['epoch'] = self.epoch
            model_dict['global_step'] = self.global_step
            model_dict['batch_size'] = self.batch_size

            torch.save(model_dict, filename)

    def training_step(self, batch):
        self.nfc.train()

        images = batch['image'].to(self.device)
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        flame = batch['flame']
        arcface = batch['arcface']
        arcface = arcface.view(-1, arcface.shape[-3], arcface.shape[-2], arcface.shape[-1]).to(self.device)
        
        inputs = {
            'images': images,
            'dataset': batch['dataset'][0]
        }

        encoder_output = self.nfc.encode(images, arcface)
        encoder_output['flame'] = flame

        # Bernardo
        imagename = batch['imagename']
        encoder_output['imagename'] = imagename
        # print('training_step - imagename:', imagename)
        imagelabel = datasets.get_imagelabel_from_imagename(imagename, self.labels_map)
        # print('training_step - imagelabel:', imagelabel)
        # y_true = batch['y_true'].to(self.device)
        # y_true = self.get_onehotvector_from_imagelabel1(imagelabel, len(list(self.labels_map.keys())))
        y_true = datasets.get_onehotvector_from_imagelabel2(imagelabel, len(list(self.labels_map.keys()))).to(self.device)
        # print('training_step - y_true:', y_true)
        encoder_output['y_true'] = y_true

        decoder_output = self.nfc.decode(encoder_output, self.epoch)
        # losses = self.nfc.compute_losses(inputs, encoder_output, decoder_output)           # original
        # losses = self.nfc.compute_losses(self.cfg, inputs, encoder_output, decoder_output)   # Bernardo
        losses, metrics = self.nfc.compute_losses(self.cfg, inputs, encoder_output, decoder_output)   # Bernardo

        all_loss = 0.
        losses_key = losses.keys()

        for key in losses_key:
            all_loss = all_loss + losses[key]

        losses['all_loss'] = all_loss

        opdict = \
            {
                'images': images,
                'flame_verts_shape': decoder_output['flame_verts_shape'],
                'pred_canonical_shape_vertices': decoder_output['pred_canonical_shape_vertices'],
                'y_pred': decoder_output['y_pred'],
                'y_true': decoder_output['y_true'],
                'metrics': metrics,
            }

        if 'deca' in decoder_output:
            opdict['deca'] = decoder_output['deca']

        return losses, opdict

    def validation_step(self):
        average_loss, smoothed_loss, avg_acc_fr = self.validator.run()
        return average_loss, smoothed_loss, avg_acc_fr

    def evaluation_step(self):
        pass

    def prepare_data(self):
        generator = torch.Generator()
        generator.manual_seed(self.device)

        # self.train_dataset, total_images = datasets.build_train(self.cfg.dataset, self.device)                            # original
        self.train_dataset, total_images = datasets.build_train_multitask_facerecognition(self.cfg.dataset, self.device)    # Bernardo

        # Bernardo
        self.labels_map = datasets.get_labels_map(self.train_dataset, self.validator.val_dataset)
        self.validator.set_labels_map(self.labels_map)
        print('prepare_data - self.labels_map:', len(list(self.labels_map.keys())), 'actors')

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            shuffle=True,
            pin_memory=True,
            # drop_last=False,    # original
            drop_last=True,       # Bernardo  -  Ignore last batch when it contains less samples than batch_size (https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274/2)
            worker_init_fn=seed_worker,
            generator=generator)

        self.train_iter = iter(self.train_dataloader)
        logger.info(f'[TRAINER] Training dataset is ready with {len(self.train_dataset)} actors and {total_images} images.')

    # Bernardo
    def get_confusion_matrix(self, y_true, y_pred, num_classes):
        assert y_true.shape == y_pred.shape
        cfm = np.zeros(shape=(self.cfg.model.num_classes, self.cfg.model.num_classes), dtype=np.float32)
        for yt, yp in zip(y_true, y_pred):
            # print('get_confusion_matrix - yt:', yt, '  yp:', yp)
            cfm[yt, yp] += 1
        return cfm

    # Bernardo
    def build_confusion_matrix_figure(self, cf_matrix, num_classes):
        scale_factor =   0.01
        # scale_factor = 0.015
        fig, ax = plt.subplots(1, 1, figsize=(num_classes*scale_factor, num_classes*scale_factor))
        im = ax.imshow(cf_matrix)

        ax.set_title(f'Confusion Matrix - num_classes: {num_classes}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xlim(-1, num_classes)
        ax.set_ylim(-1, num_classes)
        fig.tight_layout()
        return fig

    def get_angle_between_vectors_degree(self, a, b):
        # a = torch.nn.functional.normalize(torch.unsqueeze(a, 0))
        # b = torch.nn.functional.normalize(torch.unsqueeze(b, 0))
        a = torch.unsqueeze(a, 0)
        b = torch.unsqueeze(b, 0)
        inner_product = (a * b).sum(dim=1)
        a_norm = a.pow(2).sum(dim=1).pow(0.5)
        b_norm = b.pow(2).sum(dim=1).pow(0.5)
        cos = inner_product / (a_norm * b_norm)
        angle = torch.rad2deg(torch.acos(cos))
        return angle, a_norm, b_norm

    # Bernardo
    def evaluate_gradients(self, losses, global_step):
        with torch.no_grad():
            named_parameters = {}
            for name, param in self.nfc.arcface.named_parameters():
                named_parameters[name] = param

            # print('self.nfc.arcface.named_parameters()')
            grad_class_loss_wrt_features_weight =                      torch.autograd.grad(losses['class_loss'],                      named_parameters['features.weight'], retain_graph=True, create_graph=True)[0]
            grad_pred_verts_shape_canonical_diff_wrt_features_weight = torch.autograd.grad(losses['pred_verts_shape_canonical_diff'], named_parameters['features.weight'], retain_graph=True, create_graph=True)[0]
            angle_features_weight, norm_grad_class_loss_wrt_features_weight, norm_grad_pred_verts_shape_canonical_diff_wrt_features_weight  = self.get_angle_between_vectors_degree(grad_class_loss_wrt_features_weight, grad_pred_verts_shape_canonical_diff_wrt_features_weight)
            # print('    angle_features_weight:', angle_features_weight)

            grad_class_loss_wrt_features_bias =                      torch.autograd.grad(losses['class_loss'],                      named_parameters['features.bias'], retain_graph=True, create_graph=True)[0]
            grad_pred_verts_shape_canonical_diff_wrt_features_bias = torch.autograd.grad(losses['pred_verts_shape_canonical_diff'], named_parameters['features.bias'], retain_graph=True, create_graph=True)[0]
            angle_features_bias, norm_grad_class_loss_wrt_features_bias, norm_grad_pred_verts_shape_canonical_diff_wrt_features_bias  = self.get_angle_between_vectors_degree(grad_class_loss_wrt_features_bias, grad_pred_verts_shape_canonical_diff_wrt_features_bias)
            # print('    angle_features_bias:', angle_features_bias)

            self.writer.add_scalar('train_gradients/angle_grad_losses_WRT_ArcFace_features.weight:', angle_features_weight, global_step=global_step)
            self.writer.add_scalar('train_gradients/norm_grad_class_loss_WRT_ArcFace_features.weight:', norm_grad_class_loss_wrt_features_weight, global_step=global_step)
            self.writer.add_scalar('train_gradients/norm_grad_pred_verts_shape_canonical_diff_WRT_ArcFace_features.weight:', norm_grad_pred_verts_shape_canonical_diff_wrt_features_weight, global_step=global_step)
            self.writer.add_scalar('train_gradients/angle_grad_losses_WRT_ArcFace_features.bias:', angle_features_bias, global_step=global_step)
            self.writer.add_scalar('train_gradients/norm_grad_class_loss_WRT_ArcFace_features.bias:', norm_grad_class_loss_wrt_features_bias, global_step=global_step)
            self.writer.add_scalar('train_gradients/norm_grad_pred_verts_shape_canonical_diff_WRT_ArcFace_features.bias:', norm_grad_pred_verts_shape_canonical_diff_wrt_features_bias, global_step=global_step)



    # Bernardo
    def compute_save_affinity_score(self, losses_history, global_step):
        keys = list(losses_history.keys())
        if len(losses_history[keys[0]]) > 10:
            for k, v_list in losses_history.items():
                # aff_score = 1 - (v_list[-1] / (v_list[-2] + 1e-6))
                aff_score = 1 - (v_list[-1] / (v_list[-10] + 1e-6))
                self.writer.add_scalar('train_affinity_score/' + k, aff_score, global_step=self.global_step)
                # logger.info(f'  train_affinity_score/{k}: {aff_score:.4f}')
                print(f'  train_affinity_score/{k}: {aff_score:.4f}')


    def fit(self):
        self.prepare_data()
        iters_every_epoch = int(len(self.train_dataset) / self.batch_size)

        # max_epochs = int(self.cfg.train.max_steps / iters_every_epoch)     # original
        # start_epoch = self.epoch                                           # original
        max_epochs = int(self.cfg.train.max_steps / (self.batch_size * 10))  # Bernardo
        start_epoch = 0                                                      # Bernardo

        # For early stopping (Bernardo)
        all_val_average_loss_step, all_val_smoothed_loss_step, all_val_avg_acc_fr_step = [], [], []

        # Bernardo (affinity score)
        self.train_losses_history = {'pred_verts_shape_canonical_diff': [], 'class_loss': [], 'all_loss': []}

        # for epoch in range(start_epoch, max_epochs):   # original
        while not self.early_stop:                       # Bernardo

            # Bernardo
            self.cf_matrix_epoch_train = np.zeros(shape=(self.cfg.model.num_classes, self.cfg.model.num_classes), dtype=np.float32)

            # print('trainer.py: Trainer: fit(): epoch:', epoch, '    iters_every_epoch:', iters_every_epoch)   # Bernardo
            # for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch + 1}/{max_epochs}]"):
            for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{self.epoch}]"):
                # if self.global_step > self.cfg.train.max_steps:   # Original (commented by Bernardo)
                #     break                                         # Original (commented by Bernardo) 
                try:
                    batch = next(self.train_iter)
                except Exception as e:
                    self.train_iter = iter(self.train_dataloader)
                    batch = next(self.train_iter)

                visualizeTraining = self.global_step % self.cfg.train.vis_steps == 0

                self.opt.zero_grad()
                losses, opdict = self.training_step(batch)

                # GET CONFUSION MATRIX - Bernardo
                y_true = torch.argmax(opdict['y_true'], dim=1).cpu().numpy()
                y_pred = opdict['y_pred'].cpu().numpy()
                cf_matrix_batch = self.get_confusion_matrix(y_true, y_pred, self.cfg.model.num_classes)
                self.cf_matrix_epoch_train += cf_matrix_batch

                all_loss = losses['all_loss']

                # all_loss.backward()   # original
                if self.cfg.train.loss_mode == 'sum_all':
                    # all_loss.backward()
                    all_loss.backward(retain_graph=True)   # FOR GRADIENT TESTS
                elif self.cfg.train.loss_mode == 'separate':
                    if self.cfg.train.train_reconstruction:
                        losses['pred_verts_shape_canonical_diff'].backward(retain_graph=True)
                    if self.cfg.train.train_recognition:
                        losses['class_loss'].backward(retain_graph=True)
                    # for key in losses.keys():
                    #     losses[key].backward(retain_graph=True)     # Bernardo (experimenting train each loss separate)

                # Bernardo: print gradient angles and norms
                if self.cfg.train.compute_gradient_angles:
                    self.evaluate_gradients(losses, self.global_step)

                self.opt.step()
                self.global_step += 1

                if self.global_step % self.cfg.train.log_steps == 0 and self.device == 0:
                    loss_info = f"\n" \
                                f"  Epoch: {self.epoch}\n" \
                                f"  Global step: {self.global_step}\n" \
                                f"  Iter: {step}/{iters_every_epoch}\n" \
                                f"  LR: {self.opt.param_groups[0]['lr']}\n" \
                                f"  arcface_lr: {self.opt.param_groups[1]['lr']}\n" \
                                f"  face_recog_lr: {self.opt.param_groups[2]['lr']}\n" \
                                f"  Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}\n"
                    for k, v in losses.items():
                        loss_info = loss_info + f'  {k}: {v:.4f}\n'
                        if self.cfg.train.write_summary:
                            self.writer.add_scalar('train_loss/' + k, v, global_step=self.global_step)
                            self.train_losses_history[k].append(float(v.detach().cpu().numpy()))

                    # Bernardo
                    train_acc = opdict['metrics']['acc']
                    loss_info = loss_info + f'  acc: {train_acc:.4f}\n'
                    loss_info = loss_info + f'  self.cfg.output_dir: {self.cfg.output_dir}\n'
                    self.writer.add_scalar('train_loss/acc:', train_acc, global_step=self.global_step)

                    logger.info(loss_info)


                if visualizeTraining and self.device == 0:
                    visdict = {
                        'input_images': opdict['images'],
                    }
                    # add images to tensorboard
                    for k, v in visdict.items():
                        self.writer.add_images(k, np.clip(v.detach().cpu(), 0.0, 1.0), self.global_step)

                    # SAVE CONFUSION MATRIX - Bernardo
                    if self.cfg.train.compute_confusion_matrix:
                        cf_matrix_epoch_train_fig = self.build_confusion_matrix_figure(self.cf_matrix_epoch_train, self.cfg.model.num_classes)
                        self.writer.add_figure('cf_matrix_epoch_train', cf_matrix_epoch_train_fig, self.global_step)
                        plt.close()

                    pred_canonical_shape_vertices = torch.empty(0, 3, 512, 512).cuda()
                    flame_verts_shape = torch.empty(0, 3, 512, 512).cuda()
                    deca_images = torch.empty(0, 3, 512, 512).cuda()
                    input_images = torch.empty(0, 3, 224, 224).cuda()
                    L = opdict['pred_canonical_shape_vertices'].shape[0]
                    S = 4 if L > 4 else L
                    for n in np.random.choice(range(L), size=S, replace=False):
                        rendering = self.nfc.render.render_mesh(opdict['pred_canonical_shape_vertices'][n:n + 1, ...])
                        pred_canonical_shape_vertices = torch.cat([pred_canonical_shape_vertices, rendering])
                        rendering = self.nfc.render.render_mesh(opdict['flame_verts_shape'][n:n + 1, ...])
                        flame_verts_shape = torch.cat([flame_verts_shape, rendering])
                        input_images = torch.cat([input_images, opdict['images'][n:n + 1, ...]])
                        if 'deca' in opdict:
                            deca = self.nfc.render.render_mesh(opdict['deca'][n:n + 1, ...])
                            deca_images = torch.cat([deca_images, deca])

                    visdict = {}

                    if 'deca' in opdict:
                        visdict['deca'] = deca_images

                    visdict["pred_canonical_shape_vertices"] = pred_canonical_shape_vertices
                    visdict["flame_verts_shape"] = flame_verts_shape
                    visdict["images"] = input_images

                    savepath = os.path.join(self.cfg.output_dir, 'train_images/train_' + str(self.epoch) + '.jpg')
                    util.visualize_grid(visdict, savepath, size=512)

                if self.global_step % self.cfg.train.val_steps == 0:
                    # self.validation_step()   # original
                    val_average_loss_step, val_smoothed_loss_step, val_avg_acc_fr_step = self.validation_step()   # Bernardo
                    all_val_average_loss_step.append(val_average_loss_step)
                    all_val_smoothed_loss_step.append(val_smoothed_loss_step)
                    all_val_avg_acc_fr_step.append(val_avg_acc_fr_step)

                    # Bernardo (affinity score)
                    # Efficiently Identifying Task Groupings for Multi-Task Learning
                    # https://proceedings.neurips.cc/paper_files/paper/2021/file/e77910ebb93b511588557806310f78f1-Paper.pdf
                    if self.cfg.train.compute_affinity_score:
                        self.compute_save_affinity_score(self.train_losses_history, self.global_step)

                    # # early stopping (Bernardo)
                    # std_val_average_loss_step = np.std(np.array(all_val_average_loss_step[-self.early_stop_patience:]))
                    # mean_val_average_loss_step = np.mean(np.array(all_val_average_loss_step[-self.early_stop_patience:]))
                    # std_over_mean = std_val_average_loss_step / mean_val_average_loss_step
                    # print(f'all_val_average_loss_step: {all_val_average_loss_step}')
                    # print(f'std_val_average_loss_step: {std_val_average_loss_step}')
                    # print(f'mean_val_average_loss_step: {mean_val_average_loss_step}')
                    # print(f'std_over_mean: {std_over_mean}')
                    # print(f'self.early_stop_tolerance: {self.early_stop_tolerance}')
                    # print()
                    # if len(all_val_average_loss_step) >= self.early_stop_patience:
                    #     if std_over_mean <= self.early_stop_tolerance:
                    #         self.early_stop = True
                    #         print('Early stop!')

                if self.global_step % self.cfg.train.lr_update_step == 0:
                    self.scheduler.step()

                if self.global_step % self.cfg.train.eval_steps == 0:
                    self.evaluation_step()

                if self.global_step % self.cfg.train.checkpoint_steps == 0:
                    self.save_checkpoint(os.path.join(self.cfg.output_dir, 'model' + '.tar'))

                if self.global_step % self.cfg.train.checkpoint_epochs_steps == 0:
                    self.save_checkpoint(os.path.join(self.cfg.output_dir, 'model_' + str(self.global_step) + '.tar'))

            # # early stopping (Bernardo)
            # if self.epoch > 0:
            #     if val_average_loss_epoch >= last_val_average_loss_epoch + 0.1:
            #         self.early_stop = True
            #         print('Early stop!')
            # print(f'val_average_loss_epoch: {val_average_loss_epoch},   last_val_average_loss_epoch: {last_val_average_loss_epoch}')
            # last_val_average_loss_epoch = val_average_loss_epoch

            self.epoch += 1

        self.save_checkpoint(os.path.join(self.cfg.output_dir, 'model' + '.tar'))
        logger.info(f'[TRAINER] Fitting has ended! - epoch: {self.epoch}')
