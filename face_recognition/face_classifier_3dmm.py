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
import sys

# sys.path.append("./nfclib")

import torch
import torch.nn.functional as F

# from models.arcface import Arcface
# from models.generator import Generator
# from micalib.base_model import BaseModel

from loguru import logger


class FaceClassifier1_MLP(torch.nn.Module):
    def __init__(self, num_classes=None, model_cfg=None, device=None, tag='FaceClassifier1_MLP'):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(300, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        logits = self.layers(x)
        prob_pred = torch.nn.functional.softmax(logits, dim=1)
        y_pred = torch.argmax(prob_pred, dim=1)
        # return logits, prob_pred, y_pred
        # return prob_pred, y_pred
        return logits
        # return prob_pred
        # return y_pred

    def cross_entropy_loss2(self, logits_pred, y_true):
        output = -torch.mean(torch.sum(F.log_softmax(logits_pred, dim=-1) * y_true, dim=1))
        return output

    def loss_function(self):
        # return cross_entropy_loss2()
        pass


'''
class MICAMultitaskFacerecognition1(BaseModel):
    def __init__(self, config=None, device=None, tag='MICAMultitaskFacerecognition1'):
        super(MICAMultitaskFacerecognition1, self).__init__(config, device, tag)

        self.initialize()



    def create_model(self, model_cfg):
        mapping_layers = model_cfg.mapping_layers
        pretrained_path = None
        if not model_cfg.use_pretrained:
            pretrained_path = model_cfg.arcface_pretrained_model
        self.arcface = Arcface(pretrained_path=pretrained_path).to(self.device)
        self.flameModel = Generator(512, 300, self.cfg.model.n_shape, mapping_layers, model_cfg, self.device)

        # Bernardo
        self.faceClassifier = FaceClassifier1_MLP(self.cfg.model.num_classes, model_cfg, self.device).to(self.device)


    def load_model(self):
        model_path = os.path.join(self.cfg.output_dir, 'model.tar')
        if os.path.exists(self.cfg.pretrained_model_path) and self.cfg.model.use_pretrained:
            model_path = self.cfg.pretrained_model_path
        if os.path.exists(model_path):
            logger.info(f'[{self.tag}] Trained model found. Path: {model_path} | GPU: {self.device}')
            checkpoint = torch.load(model_path)
            if 'arcface' in checkpoint:
                self.arcface.load_state_dict(checkpoint['arcface'])
            if 'flameModel' in checkpoint:
                self.flameModel.load_state_dict(checkpoint['flameModel'])
        else:
            logger.info(f'[{self.tag}] Checkpoint not available starting from scratch!')

    def model_dict(self):
        return {
            'flameModel': self.flameModel.state_dict(),
            'arcface': self.arcface.state_dict(),
            'faceClassifier': self.faceClassifier.state_dict()
        }

    def parameters_to_optimize(self):
        return [
            {'params': self.flameModel.parameters(), 'lr': self.cfg.train.lr},
            {'params': self.arcface.parameters(), 'lr': self.cfg.train.arcface_lr},
            {'params': self.faceClassifier.parameters(), 'lr': self.cfg.train.face_recog_lr}
        ]

    def encode(self, images, arcface_imgs):
        codedict = {}
        codedict['arcface'] = F.normalize(self.arcface(arcface_imgs))
        codedict['images'] = images
        return codedict

    def decode(self, codedict, epoch=0):
        self.epoch = epoch

        flame_verts_shape = None
        shapecode = None

        if not self.testing:
            flame = codedict['flame']
            shapecode = flame['shape_params'].view(-1, flame['shape_params'].shape[2])
            shapecode = shapecode.to(self.device)[:, :self.cfg.model.n_shape]
            with torch.no_grad():
                flame_verts_shape, _, _ = self.flame(shape_params=shapecode)

        identity_code = codedict['arcface']
        pred_canonical_vertices, pred_shape_code = self.flameModel(identity_code)

        # original
        # output = {
        #     'flame_verts_shape': flame_verts_shape,
        #     'flame_shape_code': shapecode,
        #     'pred_canonical_shape_vertices': pred_canonical_vertices,
        #     'pred_shape_code': pred_shape_code,
        #     'faceid': codedict['arcface']
        # }
        
        # Bernardo
        # logits_pred, prob_pred, y_pred = self.faceClassifier(identity_code)    # 2D face recognition (512)
        logits_pred, prob_pred, y_pred = self.faceClassifier(pred_shape_code)    # 3D face recognition (300)
        
        y_true = None
        if 'y_true' in codedict:
            y_true = codedict['y_true']

        # Bernardo
        output = {
            'flame_verts_shape': flame_verts_shape,
            'flame_shape_code': shapecode,
            'pred_canonical_shape_vertices': pred_canonical_vertices,
            'pred_shape_code': pred_shape_code,
            'faceid': codedict['arcface'],

            'logits_pred': logits_pred,
            'prob_pred': prob_pred,
            'y_pred': y_pred,
            'y_true': y_true
        }

        return output

    # Bernardo
    # From: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    def cross_entropy_loss1(self, y_pred, y_true):
        loss = torch.nn.CrossEntropyLoss()
        output = loss(y_pred, y_true)
        # output.backward()
        return output

    def cross_entropy_loss2(self, logits_pred, y_true):
        # # Bernardo
        # print('cross_entropy_loss2 - logits_pred.shape:', logits_pred.shape)
        # print('cross_entropy_loss2 - y_true.shape:', y_true.shape)
        output = -torch.mean(torch.sum(F.log_softmax(logits_pred, dim=-1) * y_true, dim=1))
        return output

    def compute_losses(self, configs, input, encoder_output, decoder_output):
        losses = {}

        pred_verts = decoder_output['pred_canonical_shape_vertices']
        gt_verts = decoder_output['flame_verts_shape'].detach()

        pred_verts_shape_canonical_diff = (pred_verts - gt_verts).abs()

        if self.use_mask:
            pred_verts_shape_canonical_diff *= self.vertices_mask

        # # Bernardo
        # pred_verts_shape_canonical_diff = torch.mean(torch.sqrt(torch.mean(torch.pow(pred_verts_shape_canonical_diff, 2), axis=2)), axis=1)

        # losses['pred_verts_shape_canonical_diff'] = torch.mean(pred_verts_shape_canonical_diff) * 1000.0   # Original
        losses['pred_verts_shape_canonical_diff'] = configs.train.lambda1 * torch.mean(pred_verts_shape_canonical_diff) * 1000.0     # Bernardo

        # Bernardo
        logits_pred = decoder_output['logits_pred']
        prob_pred = decoder_output['prob_pred']
        y_pred = decoder_output['y_pred']
        y_true = decoder_output['y_true']
        # losses['class_loss'] = self.cross_entropy_loss1(y_pred, y_true)
        # losses['class_loss'] = self.cross_entropy_loss2(prob_pred, y_true)
        # losses['class_loss'] = self.cross_entropy_loss2(logits_pred, y_true)
        losses['class_loss'] = configs.train.lambda2 * self.cross_entropy_loss2(logits_pred, y_true)

        return losses
'''