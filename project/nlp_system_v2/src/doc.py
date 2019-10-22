import torch
import numpy as np
import pandas as pd
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from distutils.dir_util import copy_tree

class LogoDataset(Dataset):
    """Custom Dataset for loading Logo images"""

    def __init__(self, txt_path, img_dir, transform=None):

        df = pd.read_csv(txt_path, sep=",", index_col=None)
        self.img_dir = img_dir
        self.txt_path = txt_path
        self.img_names = df['Image'].values
        self.y = df['Label'].values
        self.transform = transform
        self.label_to_idx = dict()

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)

        if self.y[index] not in self.label_to_idx:
            self.label_to_idx[self.y[index]] = len(self.label_to_idx)
        label = self.label_to_idx[self.y[index]]
        return img, label, self.img_names[index]

    def __len__(self):
        return self.y.shape[0]


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MyImageFolder(ImageFolder):

    def __getitem__(self, index):
        # print(super(MyImageFolder, self).__getitem__(index))
        # print(self.imgs[index])

        return super(MyImageFolder, self).__getitem__(index) + (self.imgs[index][0],)


def load_data(data_path, tmp_path, args, shuffle=True):

    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)

    copy_tree("data_path", "tmp_path")

    remove_tree(generated_data_path)

    custom_transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                                           transforms.ToTensor(),
                                           ])

    dataset = MyImageFolder(root=data_path,
                            transform=custom_transform)

    return DataLoader(dataset=dataset,
                      batch_size=args.batch_size,
                      shuffle=shuffle,
                      num_workers=4)
