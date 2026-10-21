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


class dvb(_Loss):
    """
    Balanced Softmax Loss
    """
    def __init__(self, num_pre_cls):
        super(dvb, self).__init__()
        self.sample_per_class = torch.tensor(num_pre_cls).cuda()

    def forward(self, input, label, reduction='mean'):
        return balanced_softmax_loss(label, input, self.sample_per_class, reduction)


def balanced_softmax_loss(labels, logits, sample_per_class, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    # spc = sample_per_class.type_as(logits)
    balance = sample_per_class/torch.sum(sample_per_class)
    balance = balance.type_as(logits)
    max_prob = balance.max().item()
    spb = balance.unsqueeze(0).expand(logits.shape[0], -1)
    # prob = balance.unsqueeze(0).expand(logits.shape[0], -1)
    prob = balance / max_prob
    weight = -prob.log() + 1
    # spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spb.log()*spb
    loss = F.cross_entropy(input=logits, target=labels, weight=weight, reduction=reduction)
    return loss


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6" 
    # print(1)
    temp = torch.randn((32, 3, 224, 224))
    l = torch.tensor([199, 30, 87, 17, 33])
    print(l)