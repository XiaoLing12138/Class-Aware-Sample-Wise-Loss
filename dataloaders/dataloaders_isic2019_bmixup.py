import json
import random

import numpy as np
from PIL import Image
from PIL import ImageFilter
from skimage import io
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import transforms


class ISICDataset(Dataset):
    def __init__(self, root_path, transform, mode):
        super(ISICDataset, self).__init__()
        self.root = root_path
        self.mode = mode

        beta = 0.999
        cls_num_list = np.array([2590, 5000, 1342, 360, 695, 51, 97, 187])
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        self.weight = []

        if mode == "train":
            datas = np.load(f'{root_path}/train_100.npy')
        elif mode == "valid":
            datas = np.load(f'{root_path}/val_100.npy')
        else:
            datas = np.load(f'{root_path}/test_100.npy')

        with open(f"{root_path}/labels.json", "r", encoding="utf-8") as f:
            self.labels = json.load(f)

        # label
        self.datas = datas.tolist()

        for item in self.datas:
            self.weight.append(per_cls_weights[self.labels[item][1]])
        self.mod_sampler = WeightedRandomSampler(weights=self.weight, num_samples=len(self.weight))
        self.transform = transform

    def __getitem__(self, item):
        img_name = self.datas[item]
        path, label = self.labels[img_name]
        img = Image.fromarray(io.imread(f"{self.root}/{path}")).convert("RGB")
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.datas)


class ISICDataloader:
    def __init__(self, batch_size=128, num_workers=9, img_resize=224, root_dir=None):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_resize = img_resize
        self.root_dir = root_dir

    def run(self, mode):
        if mode != "test":
            transform = transforms.Compose(
                [
                    transforms.Resize((self.img_resize + 40, self.img_resize + 40)),
                    transforms.RandomCrop((self.img_resize, self.img_resize)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((self.img_resize + 40, self.img_resize + 40)),
                    transforms.CenterCrop(self.img_resize),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            )

        dataset = ISICDataset(
                root_path=self.root_dir,
                transform=transform,
                mode=mode
            )

        if mode == 'train':
            loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
                sampler=dataset.mod_sampler
            )
        else:
            loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )

        return loader, dataset

