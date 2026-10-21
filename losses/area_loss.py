import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class AreaLoss(nn.Module):
    def __init__(self):
        super(AreaLoss, self).__init__()

    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, weight=self.weight)

    def calculate_eff_weight(self, train_loader, model, cls_num_list):
        train_propertype = self.get_feature_mean(train_loader, model, cls_num_list)
        train_propertype = train_propertype.detach()

        model.eval()
        train_propertype = train_propertype.cuda()
        class_num = len(cls_num_list)
        eff_all = torch.zeros(class_num).float().cuda()
        with torch.no_grad():
            for i, (input, target) in enumerate(train_loader):
                target = target.cuda()
                input = input.cuda()
                input_var = self.to_var(input, requires_grad=False)
                features, output = model(input_var)
                mu = train_propertype[target].detach()  # batch_size x d
                feature_bz = (features.detach() - mu)  # Centralization
                index = torch.unique(target)  # class subset
                index2 = target.cpu().numpy()
                eff = torch.zeros(class_num).float().cuda()

                for i in range(len(index)):  # number of class
                    index3 = torch.from_numpy(np.argwhere(index2 == index[i].item()))
                    index3 = torch.squeeze(index3)
                    feature_juzhen = feature_bz[index3].detach()
                    if feature_juzhen.dim() == 1:
                        eff[index[i]] = 1
                    else:
                        _matrixA_matrixB = torch.matmul(feature_juzhen, feature_juzhen.transpose(0, 1))
                        _matrixA_norm = torch.unsqueeze(
                            torch.sqrt(torch.mul(feature_juzhen, feature_juzhen).sum(axis=1)), 1)
                        _matrixA_matrixB_length = torch.mul(_matrixA_norm, _matrixA_norm.transpose(0, 1))
                        _matrixA_matrixB_length[_matrixA_matrixB_length == 0] = 1
                        r = torch.div(_matrixA_matrixB, _matrixA_matrixB_length)  # R
                        num = feature_juzhen.size(0)
                        a = (torch.ones(1, num).float().cuda()) / num  # a_T
                        b = (torch.ones(num, 1).float().cuda()) / num  # a
                        c = torch.matmul(torch.matmul(a, r), b).float().cuda()  # a_T R a
                        eff[index[i]] = 1 / c
                eff_all = eff_all + eff
            weights = eff_all
            weights = torch.where(weights > 0, 1 / weights, weights).detach()
            # weight
            fen_mu = torch.sum(weights)
            weights_new = weights / fen_mu
            weights_new = weights_new * class_num  # Eq.(14)
            weights_new = weights_new.detach()
            self.weight = weights_new

    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)

    def get_feature_mean(self, train_loader, model, cls_num_list):
        model.eval()
        cls_num = len(cls_num_list)
        feature_mean_end = torch.zeros(cls_num, 2048)
        with torch.no_grad():
            for i, (input, target) in enumerate(train_loader):
                target = target.cuda()
                input = input.cuda()
                input_var = self.to_var(input, requires_grad=False)
                features, output = model(input_var)
                features = features.detach()
                features = features.cpu().data.numpy()

                for out, label in zip(features, target):
                    feature_mean_end[label] = feature_mean_end[label] + out

            img_num_list_tensor = torch.tensor(cls_num_list).unsqueeze(1)
            feature_mean_end = torch.div(feature_mean_end, img_num_list_tensor).detach()
            return feature_mean_end


if __name__ == "__main__":
    alpha = torch.rand(21, 1)
    print(alpha)








