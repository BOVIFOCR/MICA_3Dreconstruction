import os
import sys

import torch
import torch.nn
import torch.nn.functional as F

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


class ArcFaceLoss(torch.nn.Module):
    def __init__(self, num_classes=None, margin=None, scale=None, model_cfg=None, device=None):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(300, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, self.num_classes),
        )
        self.W = torch.nn.Parameter(torch.Tensor(self.num_classes, self.num_classes))
        torch.nn.init.xavier_normal_(self.W)

    def forward(self, x):
        embeddings = self.layers(x)
        return embeddings

    def get_arcface_loss(self, embeddings=None, labels=None):
        cosine = self.get_cosine(embeddings) # (None, n_classes)
        mask = self.get_target_mask(labels) # (None, n_classes)
        cosine_of_target_classes = cosine[mask == 1] # (None, )
        modified_cosine_of_target_classes = self.modify_cosine_of_target_classes(
            cosine_of_target_classes
        ) # (None, )
        diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(1) # (None,1)
        logits = cosine + (mask * diff) # (None, n_classes)
        logits = self.scale_logits(logits) # (None, n_classes)
        return torch.nn.CrossEntropyLoss()(logits, labels)

    def get_cosine(self, embeddings):
        """
        Args:
            embeddings: (None, embedding_size)
        Returns:
            cosine: (None, n_classes)
        """
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.W))
        return cosine

    def get_target_mask(self, labels):
        """
        Args:
            labels: (None,)
        Returns:
            mask: (None, n_classes)
        """
        batch_size = labels.size(0)
        onehot = torch.zeros(batch_size, self.num_classes, device=labels.device)
        onehot.scatter_(1, labels.unsqueeze(-1), 1)
        return onehot

    def modify_cosine_of_target_classes(self, cosine_of_target_classes):
        """
        Args:
            cosine_of_target_classes: (None,)
        Returns:
            modified_cosine_of_target_classes: (None,)
        """
        eps = 1e-6
        # theta in the paper
        angles = torch.acos(torch.clamp(cosine_of_target_classes, -1 + eps, 1 - eps))
        return torch.cos(angles + self.margin)

    def scale_logits(self, logits):
        """
        Args:
            logits: (None, n_classes)
        Returns:
            scaled_logits: (None, n_classes)
        """
        return logits * self.scale
