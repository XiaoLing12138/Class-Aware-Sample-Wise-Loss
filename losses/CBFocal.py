import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class CBFocalLoss(nn.Module):
    def __init__(self, samples_per_class, beta=0.999, fl_gamma=2):
        super(CBFocalLoss, self).__init__()
        self.beta = beta
        self.fl_gamma = fl_gamma
        self.samples_per_class = samples_per_class

    def forward(self, logits, labels):
        num_classes = logits.size(1)
        labels_one_hot = F.one_hot(labels, num_classes).float()

        effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
        effective_num_classes = np.sum(effective_num != 0)
        effective_num[effective_num == 0] = np.inf

        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * effective_num_classes
        weights = torch.tensor(weights, device=logits.device).float()

        bc_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, reduction="none")

        if self.fl_gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.fl_gamma * labels_one_hot * logits - self.fl_gamma * torch.log(1 + torch.exp(-1.0 * logits)))

        loss = modulator * bc_loss

        weighted_loss = weights * loss
        focal_loss = torch.sum(weighted_loss)

        focal_loss /= torch.sum(labels)

        return focal_loss

