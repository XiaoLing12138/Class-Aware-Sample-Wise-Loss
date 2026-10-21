import torch
import torch.nn as nn
import torch.nn.functional as F


class SEQLoss(nn.Module):
    def __init__(self, num_classes, freq_info, gamma=0.9, _lambda=5e-3):
        super(SEQLoss, self).__init__()
        '''
        param info:
            num_instances: number of samples
            num_classes: number of classes
            freq_info: [list] frequency of each category in the dataset
            _lambda: threshold to control the tail categories
            gamma: random variable to control the beta

        '''
        self.n_c = num_classes
        self._freq_info = freq_info
        self._lambda = _lambda
        self.gamma = gamma

    def exclude_func(self):
        '''
        this func is to set the beta as 1 (or 0) with probability lambda (or 1 - lambda)

        return:
            value 0 or 1
        '''
        weight = torch.zeros((self.n_c), dtype=torch.float).cuda()
        beta = torch.zeros_like(weight).uniform_()
        weight[beta < self.gamma] = 1
        weight = weight.view(1, self.n_c).expand(self.n_i, self.n_c)
        return weight

    def threshold_func(self):
        '''
        threshold func for frequancy of each category
        return:
            0 or 1
        '''
        weight = torch.zeros((self.n_c), dtype=torch.float).cuda()
        weight[self.freq_info < self._lambda] = 1
        weight = weight.view(1, self.n_c).expand(self.n_i, self.n_c)
        return weight

    def weighted_softmax(self, weight, proba):
        return torch.sum(weight * proba, dim=1).cuda()

    def forward(self, x, target):
        self.n_i = x.shape[0]
        self.freq_info = torch.tensor(self._freq_info).cuda()

        def expand_labels(preds, labels):
            target = torch.zeros_like(preds, dtype=torch.float)
            target[torch.arange(self.n_i), labels] = 1
            return target

        y_t = expand_labels(x, target)
        eql_weight = 1 - self.exclude_func() * self.threshold_func() * (1 - y_t)

        x = torch.log(x / self.weighted_softmax(eql_weight, x).unsqueeze(1).repeat(1, x.shape[1]))
        x = torch.nan_to_num(x, nan=0.0)
        res = F.nll_loss(x, target)
        return res
