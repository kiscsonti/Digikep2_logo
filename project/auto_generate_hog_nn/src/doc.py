import torch
import numpy as np
import pandas as pd
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from distutils.dir_util import copy_tree, remove_tree
import time
from skimage.feature import hog

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


def organize_files(data_path, tmp_path, args):
    while (True):

        data_directories = list()
        for root, dirs, files in os.walk(data_path):
            for d in dirs:
                data_directories.append(os.path.join(root, d))

        if len(data_directories) < args.number_of_labels:
            time.sleep(1)
            continue

        tmp_directories = list()
        for root, dirs, files in os.walk(tmp_path):
            for d in dirs:
                tmp_directories.append(os.path.join(root, d))

        nmb_of_files = dict()
        for d in data_directories:
            for root, dirs, files in os.walk(d):
                nmb_of_files[d] = len(files)

        fail = False
        for val in nmb_of_files.values():
            if val < args.batch_size:
                time.sleep(1)
                fail = True

        if fail:
            continue

        if not os.path.exists(data_path):
            time.sleep(3)
            continue
        copy_tree(data_path, tmp_path)
        remove_tree(data_path)

        nmb_of_files = dict()
        for d in tmp_directories:
            for root, dirs, files in os.walk(d):
                nmb_of_files[d] = len(files)

        fail = False
        for val in nmb_of_files.values():
            if val < 8 * args.batch_size:
                fail = True

        if fail:
            continue

        break


def load_data(data_path, tmp_path, args, shuffle=True):
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    remove_tree(tmp_path)
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)

    organize_files(data_path, tmp_path, args)

    custom_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                           transforms.Resize((args.img_size, args.img_size)),
                                           HogFeatures(),
                                           NumpyToTensor(),
                                           ])

    dataset = MyImageFolder(root=tmp_path,
                            transform=custom_transform)

    print(dataset.class_to_idx)

    return DataLoader(dataset=dataset,
                      batch_size=args.batch_size,
                      shuffle=shuffle,
                      num_workers=4)


class HogFeatures(object):

    def __init__(self, orient=9, pixels_per_cell=(1, 1), cells_per_block=(2, 2), multichannel=False):
        self.orient = orient
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.multichannel = multichannel

    def __call__(self, img):
        return hog(img, orientations=self.orient, pixels_per_cell=self.pixels_per_cell,
                   cells_per_block=self.cells_per_block, visualize=False, multichannel=self.multichannel)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NumpyToTensor(object):

    def __call__(self, img):
        return torch.from_numpy(img).float()

    def __repr__(self):
        return self.__class__.__name__ + '()'
