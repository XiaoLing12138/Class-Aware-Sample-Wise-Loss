import torch
import torch.nn as nn


class TauNorm(nn.Module):
    def __init__(self, tau=0.5):
        super(TauNorm, self).__init__()
        self.tau = tau

    def forward(self, x):
        return x / torch.pow(torch.norm(x, p=2, dim=1, keepdim=True), self.tau)
