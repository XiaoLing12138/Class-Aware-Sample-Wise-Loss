import os
import csv

import yaml
import random
import shutil

import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from utils.utils import *
from utils.metrics import *
from utils.model_initial import *

from losses.mixup import *


def train(model, train_loader, epoch, criterion, cri_weight, optimizer):
    global la, scl
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_index = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batch_index = batch_idx
        if config["model"]["loss_file_1"] in ["bclloss", "proco"]:
            inputs = torch.cat([inputs, inputs, inputs], dim=0)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs), Variable(targets)

        if config["model"]["file"] == "ResLT":
            logitH, logitM, logitT = model(inputs)
            outputs = (logitH + logitM + logitT)
        elif config["model"]["loss_file_1"] in ["mixup", "bmixup"]:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, True)
            inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
            outputs = model(inputs)
        elif config["model"]["loss_file_1"] in ["bclloss", "proco"]:
            batch_size = targets.shape[0]
            feat_mlp, logits, centers = model(inputs)
            centers = centers[:config["train"]["num_classes"]]
            _, f2, f3 = torch.split(feat_mlp, [batch_size, batch_size, batch_size], dim=0)
            features = torch.cat([f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
            outputs, _, __ = torch.split(logits, [batch_size, batch_size, batch_size], dim=0)
        elif config["model"]["loss_file_1"] == "area_loss":
            features, outputs = model(inputs)
        elif config["model"]["loss_file_1"] == "SCLloss":
            outputs, features = model(inputs)
        else:
            outputs = model(inputs)

        if config["train"]["mixed_loss"]:
            if config["model"]["loss_file_1"] == "proco":
                contrast_logits1 = criterion[0](f2, targets)
                contrast_logits2 = criterion[0](f3, targets)
                loss = criterion[1](outputs, targets) + cri_weight[0] * (criterion[1](contrast_logits1, targets) + criterion[1](contrast_logits2, targets)) / 2
            elif config["model"]["loss_file_1"] == "bclloss":
                loss = cri_weight[0] * criterion[0](centers, features, targets) + cri_weight[1] * criterion[1](outputs, targets)
            elif config["model"]["loss_file_1"] == "area_loss" and epoch <= 40:
                loss = F.cross_entropy(outputs, targets)
            elif config["model"]["loss_file_1"] == "SCLloss":
                loss = cri_weight[0] * criterion[0](torch.stack([outputs, features], dim=1), targets)
            else:
                loss = cri_weight[0] * criterion[0](outputs, targets) + cri_weight[1] * criterion[1](outputs, targets)
        else:
            if config["model"]["loss_file_1"] in ["mixup", "bmixup"]:
                mixup_func = mixup_criterion(targets_a, targets_b, lam)
                loss = mixup_func(criterion[0], outputs)
            elif config["model"]["file"] == "LA":
                outputs = outputs + la.adjustment.cuda()
                loss = mixup_func(criterion[0], outputs)
            elif config["model"]["loss_file_1"] == "SCLloss" and scl is None:
                loss = cri_weight[0] * criterion[0](torch.stack([outputs, features], dim=1), targets)
            else:
                loss = cri_weight[0] * criterion[0](outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct = correct.item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return train_loss / batch_index, 100. * correct / total


def sub_train(model, lws, train_loader, epoch, criterion, cri_weight, optimizer):
    global crt
    print('\nEpoch: %d' % epoch)
    model.eval()
    if lws is not None:
        lws.train()
    elif crt is not None:
        model.classifier.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_index = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batch_index = batch_idx
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)
        if lws is not None:
            outputs = lws(outputs)
        elif crt is not None:
            outputs = crt(outputs)
        if config["train"]["mixed_loss"]:
            loss = cri_weight[0] * criterion[0](outputs, targets) + cri_weight[1] * criterion[1](outputs, targets)
        else:
            loss = cri_weight[0] * criterion[0](outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct = correct.item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return train_loss / batch_index, 100. * correct / total


def valid(model, valid_loader, epoch, criterion, cri_weight):
    global best_acc, best_auc, best_ece, best_bs, config, la, scl
    model.eval()
    valid_loss = 0
    correct = 0
    total = 0
    y_true = None
    y_pred = None
    labels = []
    pre_labels = []
    auc_output = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            if config["model"]["loss_file_1"] in ["bclloss", "proco"]:
                inputs = torch.cat([inputs, inputs, inputs], dim=0)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            inputs, targets = Variable(inputs), Variable(targets)

            if config["model"]["file"] == "ResLT":
                logitH, logitM, logitT = model(inputs)
                outputs = (logitH + logitM + logitT)
            elif config["model"]["loss_file_1"] in ["mixup", "bmixup"]:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, True)
                inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
                outputs = model(inputs)
            elif config["model"]["loss_file_1"] in ["bclloss", "proco"]:
                batch_size = targets.shape[0]
                feat_mlp, logits, centers = model(inputs)
                centers = centers[:config["train"]["num_classes"]]
                _, f2, f3 = torch.split(feat_mlp, [batch_size, batch_size, batch_size], dim=0)
                features = torch.cat([f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
                outputs, _, __ = torch.split(logits, [batch_size, batch_size, batch_size], dim=0)
            elif config["model"]["loss_file_1"] == "area_loss":
                _, outputs = model(inputs)
            elif config["model"]["loss_file_1"] == "SCLloss":
                outputs, features = model(inputs)
            else:
                outputs = model(inputs)

            if config["train"]["mixed_loss"]:
                if config["model"]["loss_file_1"] == "proco":
                    contrast_logits1 = criterion[0](f2, targets)
                    contrast_logits2 = criterion[0](f3, targets)
                    loss = criterion[1](outputs, targets) + cri_weight[0] * (
                                criterion[1](contrast_logits1, targets) + criterion[1](contrast_logits2, targets)) / 2
                elif config["model"]["loss_file_1"] == "bclloss":
                    loss = cri_weight[0] * criterion[0](centers, features, targets) + cri_weight[1] * criterion[1](outputs, targets)
                elif config["model"]["loss_file_1"] == "area_loss":
                    loss = F.cross_entropy(outputs, targets)
                elif config["model"]["loss_file_1"] == "SCLloss":
                    loss = cri_weight[0] * criterion[0](torch.stack([outputs, features], dim=1), targets)
                else:
                    loss = cri_weight[0] * criterion[0](outputs, targets) + cri_weight[1] * criterion[1](outputs,
                                                                                                         targets)
            else:
                if config["model"]["loss_file_1"] in ["mixup", "bmixup"]:
                    mixup_func = mixup_criterion(targets_a, targets_b, lam)
                    loss = mixup_func(criterion[0], outputs)
                elif config["model"]["file"] == "LA":
                    outputs = outputs + la.adjustment.cuda()
                    loss = mixup_func(criterion[0], outputs)
                elif config["model"]["loss_file_1"] == "SCLloss" and scl is None:
                    loss = cri_weight[0] * criterion[0](torch.stack([outputs, features], dim=1), targets)
                else:
                    loss = cri_weight[0] * criterion[0](outputs, targets)

            auc_output.extend(outputs.data.cpu().numpy())

            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            labels.extend(targets.data.cpu())
            pre_labels.extend(predicted.data.cpu())
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()

            y_true, y_pred = assemble_labels(y_true, y_pred, targets, outputs)

            progress_bar(batch_idx, len(valid_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (valid_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total

    # if acc >= best_acc and auc_avg >= best_auc and ece <= best_ece and bs <= best_bs:
    if acc > best_acc:
        best_acc = acc
        save_checkpoint(
            model,
            acc,
            epoch,
            f'{sys.path[0]}/{config["model"]["folder"]}/{config["model"]["name"]}_{config["train"]["seed"]}_{config["train"]["lr"]}.pth'
        )

    return valid_loss / batch_idx, 100. * correct / total


def sub_valid(model, lws, valid_loader, epoch, criterion, cri_weight):
    global best_acc, best_auc, best_ece, best_bs, config
    model.eval()
    lws.eval()
    valid_loss = 0
    correct = 0
    total = 0
    y_true = None
    y_pred = None
    labels = []
    pre_labels = []
    auc_output = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            outputs = lws(outputs)
            auc_output.extend(outputs.data.cpu().numpy())
            if config["train"]["mixed_loss"]:
                loss = cri_weight[0] * criterion[0](outputs, targets) + cri_weight[1] * criterion[1](outputs, targets)
            else:
                loss = cri_weight[0] * criterion[0](outputs, targets)

            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            labels.extend(targets.data.cpu())
            pre_labels.extend(predicted.data.cpu())
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()

            y_true, y_pred = assemble_labels(y_true, y_pred, targets, outputs)

            progress_bar(batch_idx, len(valid_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (valid_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total

    # if acc >= best_acc and auc_avg >= best_auc and ece <= best_ece and bs <= best_bs:
    if acc > best_acc:
        best_acc = acc
        save_checkpoint_lws(
            model,
            lws,
            acc,
            epoch,
            f'{sys.path[0]}/{config["model"]["folder"]}/{config["model"]["name"]}_{config["train"]["seed"]}_{config["train"]["lr"]}.pth'
        )

    return valid_loss / batch_idx, 100. * correct / total


# Testing
def b_test(b_test_loader, criterion):
    global best_acc, log_file, config, distribution, lws, tau
    print("begin test")
    checkpoint = torch.load(
        f'{sys.path[0]}/{config["model"]["folder"]}/{config["model"]["name"]}_{config["train"]["seed"]}_{config["train"]["lr"]}.pth'
    )
    model = checkpoint['net']
    if lws is not None:
        lws = checkpoint["lws"]
    if tau is not None:
        model.fc.weight = torch.nn.Parameter(tau(model.fc.weight))
    best_acc = checkpoint['acc']
    epoch = checkpoint['epoch']
    torch.set_rng_state(checkpoint['rng_state'])

    if use_cuda:
        model.cuda()
        if lws is not None:
            lws.cuda()
            lws.eval()
        print('Using', torch.cuda.device_count(), 'GPUs.')
        cudnn.deterministic = True
        print('Using CUDA..')

    model.eval()
    print(epoch)
    valid_loss = 0
    correct = 0
    total = 0
    y_true = None
    y_pred = None
    labels = []
    pre_labels = []
    auc_output = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(b_test_loader):
            if config["model"]["loss_file_1"] in ["bclloss", "proco"]:
                inputs = torch.cat([inputs, inputs, inputs], dim=0)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            inputs, targets = Variable(inputs), Variable(targets)

            if config["model"]["file"] == "ResLT":
                logitH, logitM, logitT = model(inputs)
                outputs = (logitH + logitM + logitT)
            elif config["model"]["loss_file_1"] in ["bclloss", "proco"]:
                batch_size = targets.shape[0]
                feat_mlp, logits, centers = model(inputs)
                _, f2, f3 = torch.split(feat_mlp, [batch_size, batch_size, batch_size], dim=0)
                outputs, _, __ = torch.split(logits, [batch_size, batch_size, batch_size], dim=0)
            elif config["model"]["loss_file_1"] == "area_loss":
                _, outputs = model(inputs)
            elif config["model"]["loss_file_1"] == "SCLloss":
                outputs, _ = model(inputs)
            else:
                outputs = model(inputs)

            if lws is not None:
                outputs = lws(outputs)
            auc_output.extend(outputs.data.cpu().numpy())

            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            labels.extend(targets.data.cpu())
            pre_labels.extend(predicted.data.cpu())
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()

            y_true, y_pred = assemble_labels(y_true, y_pred, targets, outputs)

            progress_bar(batch_idx, len(b_test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (valid_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    labels = np.reshape(labels, [-1, 1])
    pre_labels = np.reshape(pre_labels, [-1, 1])

    y_test_binarized = label_binarize(labels, classes=np.unique(labels))
    n_classes = y_test_binarized.shape[1]

    auc_output = np.array(auc_output)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], auc_output[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    auc_list = list(roc_auc.values())

    ece = get_ece(y_pred, y_true)
    cece = get_cece(y_pred, y_true)
    bs = get_bs(y_pred, y_true)

    # label -->
    auc_labels = np.zeros(shape=(len(labels), config["train"]["num_classes"]))
    for i in range(len(labels)):
        auc_labels[i][labels[i]] = 1

    head_gt = []
    head_pr = []
    mid_gt = []
    mid_pr = []
    tail_gt = []
    tail_pr = []
    for index in range(len(labels)):
        if labels[index] in distribution["head"]:
            head_pr.append(pre_labels[index])
            head_gt.append(labels[index])
        elif labels[index] in distribution["tail"]:
            tail_pr.append(pre_labels[index])
            tail_gt.append(labels[index])
        else:
            mid_pr.append(pre_labels[index])
            mid_gt.append(labels[index])

    overall_conf = classification_report(labels, pre_labels, digits=5)
    head_conf = classification_report(head_gt, head_pr, digits=5)
    mid_conf = classification_report(mid_gt, mid_pr, digits=5)
    tail_conf = classification_report(tail_gt, tail_pr, digits=5)

    head_acc = accuracy_score(head_gt, head_pr)
    mid_acc = accuracy_score(mid_gt, mid_pr)
    tail_acc = accuracy_score(tail_gt, tail_pr)

    with open(log_file, 'a') as f:
        f.write(f"balance-------------------------------------\n")
        f.write(f"overall: \n{overall_conf}\n")
        f.write(f"head: \n{head_conf}\n")
        f.write(f"mid: \n{mid_conf}\n")
        f.write(f"tail: \n{tail_conf}\n")
        f.write(f"avg acc: \n{(head_acc+mid_acc+tail_acc)/3}\n")
        f.write(f"auc average: \n{np.mean(auc_list)}\n")
        f.write(f"auc std: \n{np.std(auc_list)}\n")
        f.write(f"ece: \n{ece.item()}\n")
        f.write(f"cece: \n{cece.item()}\n")
        f.write(f"bs: \n{bs}\n")
        f.write(f"balance-------------------------------------\n")


def prepare_task(config):
    global best_acc, best_auc, best_ece, best_bs, use_cuda, log_file, config_name, distribution, lws, tau, la, scl, crt
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['train']['cuda'])
    best_acc, best_auc, best_ece, best_bs = 0, 0, 100, 100
    lws, tau, la, scl, crt = None, None, None, None, None
    distribution = {"head": [0, 1], "mid": [2, 3, 4], "tail": [5, 6, 7]}
    # [2590, 5000, 1342, 360, 695, 51, 97, 187]
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(config["train"]["seed"])
    np.random.seed(config["train"]["seed"])
    random.seed(config["train"]["seed"])

    if not os.path.isdir(config["model"]["folder"]):
        os.makedirs(config["model"]["folder"])

    if not os.path.isdir(config["log"]["folder"]):
        os.makedirs(config["log"]["folder"])

    shutil.copyfile(
        f'{sys.path[0]}/configs/{config_name}',
        f'{sys.path[0]}/{config["log"]["folder"]}/{config["model"]["name"]}_{config["train"]["seed"]}_{config["train"]["lr"]}.log'
    )

    log_file = f'{sys.path[0]}/{config["log"]["folder"]}/{config["model"]["name"]}_{config["train"]["seed"]}_{config["train"]["lr"]}.csv'
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            log_writer = csv.writer(f, delimiter=',')
            log_writer.writerow(['epoch', 'train loss', 'train acc', 'valid loss', 'valid acc'])


def main():
    global best_acc, best_auc, best_ece, best_bs, log_file, config, config_name, lws, tau, la, scl, crt
    config_name = 'config_isic2019.yaml'
    with open(f'configs/{config_name}', 'rb') as f:
        config = yaml.safe_load(f.read())

    prepare_task(config)

    if config["model"]["loss_file_1"] == "bmixup":
        from dataloaders.dataloaders_isic2019_bmixup import ISICDataloader
        dataloader = ISICDataloader(
            batch_size=config["train"]["batch-size"],
            num_workers=config["train"]["num_worker"],
            img_resize=config["train"]["size"],
            root_dir=config["train"]["data_path"],
        )
    else:
        from dataloaders.dataloaders_isic2019 import ISICDataloader
        dataloader = ISICDataloader(
            batch_size=config["train"]["batch-size"],
            num_workers=config["train"]["num_worker"],
            img_resize=config["train"]["size"],
            root_dir=config["train"]["data_path"],
        )

    (test_loader, test_dataset,) = dataloader.run("test")
    (valid_loader, valid_dataset,) = dataloader.run("valid")
    (train_loader, train_dataset,) = dataloader.run("train")

    model = create_model(config)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.deterministic = True
    print('Using', torch.cuda.device_count(), 'GPUs.')
    print('Using CUDA..')

    cri_weight = [config["model"]["alpha"], config["model"]["beta"]]

    if config["train"]["mixed_loss"]:
        if config["model"]["loss_file_1"] not in ["None", "bmixup", "mixup", "cRT"]:
            exec(f'from losses.{config["model"]["loss_file_1"]} import *')
        if config["model"]["loss_file_2"] not in ["None", "bmixup", "mixup", "cRT"]:
            exec(f'from losses.{config["model"]["loss_file_2"]} import *')

        if config["model"]["loss_file_1"] == "Tau":
            tau = eval(f'{config["model"]["loss_caller_1"]}')
            criterion_1 = torch.nn.CrossEntropyLoss()
        elif config["model"]["loss_file_1"] == "LWS_loss":
            lws = eval(f'{config["model"]["loss_caller_1"]}')
            criterion_1 = torch.nn.CrossEntropyLoss()
        elif config["model"]["loss_file_1"] == "LA":
            la = eval(f'{config["model"]["loss_caller_1"]}')
            criterion_1 = torch.nn.CrossEntropyLoss()
        elif config["model"]["loss_file_1"] == "crt":
            crt = eval(f'{config["model"]["loss_caller_1"]}')
            criterion_1 = torch.nn.CrossEntropyLoss()
        else:
            criterion_1 = eval(f'{config["model"]["loss_caller_1"]}')

        if config["model"]["loss_file_2"] == "Tau":
            tau = eval(f'{config["model"]["loss_caller_2"]}')
            criterion_2 = torch.nn.CrossEntropyLoss()
        elif config["model"]["loss_file_2"] == "LWS_loss":
            lws = eval(f'{config["model"]["loss_caller_2"]}')
            criterion_2 = torch.nn.CrossEntropyLoss()
        elif config["model"]["loss_file_2"] == "LA":
            la = eval(f'{config["model"]["loss_caller_2"]}')
            criterion_2 = torch.nn.CrossEntropyLoss()
        elif config["model"]["loss_file_2"] == "crt":
            crt = eval(f'{config["model"]["loss_caller_2"]}')
            criterion_2 = torch.nn.CrossEntropyLoss()
        else:
            criterion_2 = eval(f'{config["model"]["loss_caller_2"]}')

        criterion = [criterion_1, criterion_2]
    else:
        if config["model"]["loss_file_1"] not in ["None", "bmixup", "mixup", "cRT"]:
            exec(f'from losses.{config["model"]["loss_file_1"]} import *')

        if config["model"]["loss_file_1"] == "Tau":
            tau = eval(f'{config["model"]["loss_caller_1"]}')
            criterion = torch.nn.CrossEntropyLoss()
        elif config["model"]["loss_file_1"] == "LWS_loss":
            lws = eval(f'{config["model"]["loss_caller_1"]}')
            criterion = torch.nn.CrossEntropyLoss()
        elif config["model"]["loss_file_1"] == "LA":
            la = eval(f'{config["model"]["loss_caller_1"]}')
            criterion = torch.nn.CrossEntropyLoss()
        elif config["model"]["loss_file_1"] == "crt":
            crt = eval(f'{config["model"]["loss_caller_1"]}')
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = eval(f'{config["model"]["loss_caller_1"]}')
        criterion = [criterion, None]

    # optimizer = optim.SGD(model.parameters(), lr=config["train"]["lr"], momentum=0.9,
    #                       weight_decay=config["train"]["decay"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])

    if config["train"]["resume"]:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(
            os.path.join(sys.path[0], config["model"]["folder"])), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(
            f'{sys.path[0]}/{config["model"]["folder"]}/{config["model"]["name"]}_{config["train"]["seed"]}_{config["train"]["lr"]}.pth')
        model = torch.nn.DataParallel(checkpoint['net'])
        start_epoch = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])
    else:
        print('==> Building model...')
        start_epoch = 0

    for epoch in range(start_epoch, config["train"]["epochs"]):
        if config["model"]["loss_file_1"] == "area_loss":
            criterion[0].calculate_eff_weight(train_loader, model, [2590, 5000, 1342, 360, 695, 51, 97, 187])

        train_loss, train_acc = train(model, train_loader, epoch, criterion, cri_weight, optimizer)
        valid_loss, valid_acc = valid(model, valid_loader, epoch, criterion, cri_weight)
        with open(log_file, 'a') as f:
            log_writer = csv.writer(f, delimiter=',')
            log_writer.writerow([epoch, train_loss, train_acc, valid_loss, valid_acc])

    if lws is not None:
        best_acc = 0
        optimizer = torch.optim.Adam(lws.parameters(), lr=config["train"]["lr"])
        checkpoint = torch.load(
            f'{sys.path[0]}/{config["model"]["folder"]}/{config["model"]["name"]}_{config["train"]["seed"]}_{config["train"]["lr"]}.pth'
        )
        model = checkpoint['net']
        model = torch.nn.DataParallel(model).cuda()
        lws = torch.nn.DataParallel(lws).cuda()
        for epoch in range(10):
            train_loss, train_acc = sub_train(model, lws, train_loader, epoch, criterion, cri_weight, optimizer)
            valid_loss, valid_acc = sub_valid(model, lws, valid_loader, epoch, criterion, cri_weight)
            with open(log_file, 'a') as f:
                log_writer = csv.writer(f, delimiter=',')
                log_writer.writerow([epoch, train_loss, train_acc, valid_loss, valid_acc])

    if crt is not None:
        best_acc = 0
        checkpoint = torch.load(
            f'{sys.path[0]}/{config["model"]["folder"]}/{config["model"]["name"]}_{config["train"]["seed"]}_{config["train"]["lr"]}.pth'
        )
        model = checkpoint['net']
        model.classifier = torch.nn.Linear(512*4, config['model']['num_classes'])
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=config["train"]["lr"])
        model = torch.nn.DataParallel(model).cuda()
        for epoch in range(10):
            train_loss, train_acc = sub_train(model, lws, train_loader, epoch, criterion, cri_weight, optimizer)
            valid_loss, valid_acc = sub_valid(model, lws, valid_loader, epoch, criterion, cri_weight)
            with open(log_file, 'a') as f:
                log_writer = csv.writer(f, delimiter=',')
                log_writer.writerow([epoch, train_loss, train_acc, valid_loss, valid_acc])

    if config["model"]["loss_file_1"] == "SCLloss":
        scl = True
        for epoch in range(50):
            train_loss, train_acc = train(model, train_loader, epoch, [torch.nn.CrossEntropyLoss(), None], cri_weight, optimizer)
            valid_loss, valid_acc = valid(model, valid_loader, epoch, [torch.nn.CrossEntropyLoss(), None], cri_weight)
            with open(log_file, 'a') as f:
                log_writer = csv.writer(f, delimiter=',')
                log_writer.writerow([epoch, train_loss, train_acc, valid_loss, valid_acc])

    b_test(test_loader, torch.nn.CrossEntropyLoss())


if __name__ == '__main__':
    print(sys.path[0])
    main()



