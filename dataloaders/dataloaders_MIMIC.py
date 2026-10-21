import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class MIMICDataset(Dataset):
    def __init__(self, root_path, mode, transform):
        super(MIMICDataset, self).__init__()
        self.root = root_path
        self.mode = mode
        self.transform = transform

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

    def __getitem__(self, item):
        path, label = self.data[item]
        img = np.load(f"{self.root}/{path[36:]}")
        img = np.stack([img, img, img], axis=-1)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


class MIMICDataloader:
    def __init__(self, batch_size=128, num_workers=9, img_resize=224, root_dir=None):
        super(MIMICDataloader, self).__init__()
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

        dataset = MIMICDataset(
            root_path=self.root_dir,
            mode=mode,
            transform=transform
        )

        if mode == 'train':
            loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
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
