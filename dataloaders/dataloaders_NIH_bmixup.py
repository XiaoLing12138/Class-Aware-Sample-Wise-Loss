import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import transforms


class NIHDataset(Dataset):
    def __init__(self, root_path, mode, transform):
        super(NIHDataset, self).__init__()
        self.root = root_path
        self.mode = mode
        self.transform = transform

        beta = 0.999
        cls_num_list = [2351, 600, 651, 326, 2156, 171, 389, 36, 5205, 1248, 1527, 608, 159, 1113, 44, 5, 24, 254, 90,
                        34079]
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        self.weight = []


        if mode == "train":
            df = pd.read_csv(f"{root_path}/train.csv")
        elif mode == "valid":
            df = pd.read_csv(f"{root_path}/b-valid.csv")
        elif mode == "b-test":
            df = pd.read_csv(f"{root_path}/b-test.csv")
        elif mode == "ub-test":
            df = pd.read_csv(f"{root_path}/ub-test.csv")
        else:
            raise Exception

        self.data = df.values.tolist()

        for item in self.data:
            self.weight.append(per_cls_weights[item[1]])
        self.mod_sampler = WeightedRandomSampler(weights=self.weight, num_samples=len(self.weight))

    def __getitem__(self, item):
        path, label = self.data[item]
        img = np.load(f"{self.root}/{path[33:]}")
        img = np.stack([img, img, img], axis=-1)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


class NIHDataloader:
    def __init__(self, batch_size=128, num_workers=9, img_resize=224, root_dir=None):
        super(NIHDataloader, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_resize = img_resize
        self.root_dir = root_dir

    def run(self, mode):
        if mode == 'train':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

        dataset = NIHDataset(
            root_path=self.root_dir,
            mode=mode,
            transform=transform
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
