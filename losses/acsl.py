import torch
import torch.nn as nn
import torch.nn.functional as F


class ACSL(nn.Module):

    def __init__(self, score_thr=0.7, loss_weight=1.0):

        super(ACSL, self).__init__()

        self.score_thr = score_thr
        self.loss_weight = loss_weight

    def forward(self, cls_logits, labels):
        device = cls_logits.device
        self.n_i, self.n_c = cls_logits.size()

        sigmoid_cls_logits = torch.sigmoid(cls_logits)

        high_score_inds = torch.nonzero(sigmoid_cls_logits >= self.score_thr)
        weight_mask = torch.sparse_coo_tensor(high_score_inds.t(), cls_logits.new_ones(high_score_inds.shape[0]),
                                              size=(self.n_i, self.n_c), device=device).to_dense()

        for i in range(self.n_i):
            weight_mask[i, labels[i]] = 1

        target = cls_logits.new_zeros(self.n_i, self.n_c)

        for cls in torch.unique(labels):
            cls = cls.item()
            cls_inds = torch.nonzero(labels == cls).squeeze(1)

            cur_labels = [cls]
            cur_labels = torch.tensor(cur_labels, device=cls_logits.device)
            tmp_label_vec = cls_logits.new_zeros(self.n_c)
            tmp_label_vec[cur_labels] = 1
            tmp_label_vec = tmp_label_vec.expand(cls_inds.numel(), self.n_c)
            target[cls_inds] = tmp_label_vec

        cls_loss = F.binary_cross_entropy_with_logits(cls_logits, target, reduction='none')

        return torch.sum(weight_mask * cls_loss) / self.n_i


if __name__ == '__main__':
    loss = ACSL()
    ce = torch.nn.CrossEntropyLoss()
    s = [[0.5, 0.1, 0.1, 0.1, 0.7],
         [0.1, 0.8, 0.7, 0.1, 0.2],
         [0.1, 0.2, 0.9, 0.4, 0.1],
         [0.1, 0.2, 0.9, 0.3, 0.1],
         [0.1, 0.7, 0.3, 0.1, 0.1]]
    r = [0, 1, 2, 3, 4]
    a = torch.tensor(s)
    b = torch.tensor(r)
    acsl = ACSL()
    t = acsl.forward(a, b)
    print(ce(a, b))
    print(t)

