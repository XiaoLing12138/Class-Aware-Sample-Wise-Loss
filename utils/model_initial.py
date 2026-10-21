import torch
import torchvision


def create_model(config):
    if config["train"]["pretrain"] == "official":
        if config["model"]["pretrain"] == "resnet50":
            exec(f'from models.{config["model"]["file"]} import *')
            model = eval(f'{config["model"]["caller"]}')
            model_dict = torch.load('resnet50-0676ba61.pth', map_location='cpu')
            model_dict.pop('fc.weight')
            model_dict.pop('fc.bias')
            if config["model"]["loss_file_1"] == "SCLloss":
                model_dict.pop('conv1.weight')
            model.load_state_dict(model_dict, strict=False)
        elif config["model"]["pretrain"] == "swin_t":
            exec(f'from models.{config["model"]["file"]} import *')
            model = eval(f'{config["model"]["caller"]}')
            model_dict = torch.load('swin_t-704ceda3.pth', map_location='cpu')
            model_dict.pop('head.weight')
            model_dict.pop('head.bias')
            model.load_state_dict(model_dict, strict=False)
    elif config["train"]["pretrain"] == "self":
        checkpoint = torch.load(config["model"]["pretrain"])
        model = checkpoint['net']
    else:
        exec(f'from models.{config["model"]["file"]} import *')
        model = eval(f'{config["model"]["caller"]}')
    return model
