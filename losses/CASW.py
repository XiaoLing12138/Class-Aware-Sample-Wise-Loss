import numpy
import torch
from torch import nn

import torch.nn.functional as F
from torch.autograd import Variable


class CASWLoss(nn.Module):

    def __init__(self):
        super(CASWLoss, self).__init__()

    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        class_num = inputs.size(1)
        predicted_values = F.softmax(inputs, 1)
        # 1 hot encoding
        y_zerohot = torch.zeros(batch_size, class_num).scatter_(1, targets.view(batch_size, 1).data.cpu(), 1)
        temp_output = torch.zeros(class_num)

        for i in range(class_num):
            if torch.sum(y_zerohot[:, i]) > 0:
                temp_output[i] = torch.sum(
                    -torch.log(predicted_values[:, i] + 1e-7) * y_zerohot[:, i].cuda()) / torch.sum(y_zerohot[:, i])

        loss = temp_output.mean()
        return loss


if __name__ == "__main__":
    alpha = torch.rand(21, 1)
    print(alpha)
    FL = MacroLoss()
    CE = nn.CrossEntropyLoss()
    N = 20
    C = 5
    inputs = torch.rand(N, C)

    targets = torch.LongTensor(N).random_(C)
    inputs_fl = Variable(inputs.clone(), requires_grad=True)
    targets_fl = Variable(targets.clone())

    inputs_ce = Variable(inputs.clone(), requires_grad=True)
    targets_ce = Variable(targets.clone())
    print('----inputs----')
    print(inputs)
    print('---target-----')
    print(targets)

    fl_loss = FL(inputs_fl, targets_fl)

    ce_loss = CE(inputs_ce, targets_ce)
    print(ce_loss, fl_loss)
    fl_loss.backward()
    ce_loss.backward()
    print(inputs_ce.grad.data)
