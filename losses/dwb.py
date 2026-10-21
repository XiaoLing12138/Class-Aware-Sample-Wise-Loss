"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""


import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import json


class DWB(_Loss):
    """
    Balanced Softmax Loss
    """
    def __init__(self, num_pre_cls):
        super(DWB, self).__init__()
        # with open(freq_path, 'r') as fd:
        #     freq = json.load(fd)
        # freq = torch.tensor(freq)
        # self.sample_per_class = freq
        self.sample_per_class = torch.tensor(num_pre_cls).cuda()
        # self.sample_per_class = torch.tensor([199,30,87,17,33]).cuda()
        # self.sample_per_class = torch.tensor([34,5,32,19,13]).cuda()
        # self.sample_per_class = torch.tensor([122,98]).cuda()

    def forward(self, input, label, reduction='mean'):
        return balanced_softmax_loss(label, input, self.sample_per_class, reduction)


def balanced_softmax_loss(labels, logits, sample_per_class, reduction):
    """Compute the Balanced Softmax Loss between logits and the ground truth labels.
    
    Args:
        labels: A int tensor of size [batch].
        logits: A float tensor of size [batch, no_of_classes].
        sample_per_class: A int tensor of size [no of classes].
        reduction: string. One of "none", "mean", "sum"
        
    Returns:
        loss: A float tensor. Balanced Softmax Loss.
    """
    # Normalize sample_per_class to get balance weights
    balance = sample_per_class.float() / sample_per_class.sum()
    
    # Calculate maximum probability for normalization
    max_prob = balance.max().item()
    prob = balance / max_prob
    
    # Convert logits to probabilities
    p_ij = F.softmax(logits, dim=1)
    
    # Calculate weights based on the new formula
    weight = (1 - prob).pow(1 - p_ij).detach()  # Ensure no gradient tracking

    # Ensure weight is a 1D tensor of shape [num_classes]
    weight = weight.mean(dim=0)  # Average the weights across the batch

    # Calculate the cross-entropy loss with the modified weight
    loss = F.cross_entropy(input=logits, target=labels, weight=weight, reduction=reduction)

    # Add the second term from the formula
    second_term = (p_ij * (1 - p_ij)).sum(dim=1)

    if reduction == 'mean':
        second_term = second_term.mean()
    
    loss += second_term

    return loss


def create_loss(freq_path):
    print('Loading Balanced Softmax Loss.')
    return BalancedSoftmax(freq_path)


if __name__ == '__main__':
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create an instance of the DWB loss
    dwb_loss = DWB().to(device)

    # Generate random logits and labels
    batch_size = 32
    num_classes = 5

    # Generate random logits (e.g., from a model output)
    logits = torch.randn(batch_size, num_classes).to(device)
    
    # Generate random labels (integers in the range [0, num_classes-1])
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)

    # Calculate the loss
    print(logits.shape)
    print(labels.shape)
    loss = dwb_loss(logits, labels, reduction='mean')

    # Print the results
    print("Logits:\n", logits)
    print("Labels:\n", labels)
    print("Calculated Loss:\n", loss.item())