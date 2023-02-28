import torch

def cross_entropy_loss_logits_and_targets():
    return torch.nn.CrossEntropyLoss()


# source: https://www.kaggle.com/code/nanguyen/arcface-loss
def arcface_loss(embeddings=None, labels=None, margin=0.5, scale=1.0):
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
