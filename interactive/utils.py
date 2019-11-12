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
import os
import time
import torch
import random
from datetime import datetime

def evaluate(dev_data, model, eval_train=False, max_input=None, use_cuda=False):
    model.eval()
    predictions = []
    actuals = []
    all_img_names = []
    good = 0
    total = 0
    nmb_of_input = 0
    for inputs, labels, image_names in dev_data:
        if max_input is not None and nmb_of_input >= max_input:
            break

        nmb_of_input += len(image_names)
        if use_cuda:
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
        pred_proba = model(inputs)

        actuals.extend(labels)
        all_img_names.extend(image_names)

        final_predicts = torch.argmax(pred_proba, dim=1)
        predictions.extend(final_predicts)
        total += len(labels)
        for a, b in zip(final_predicts, labels):
            if a == b:
                good += 1

    if eval_train:
        return good / total

    acc = good / total
    return acc

def evaluate_details(dev_data, model, eval_train=False, max_input=None, use_cuda=False):
    model.eval()
    predictions = []
    actuals = []
    all_img_names = []
    good = 0
    total = 0
    nmb_of_input = 0
    gathered_data = dict()

    for inputs, labels, image_names in dev_data:
        if max_input is not None and nmb_of_input >= max_input:
            break

        nmb_of_input += len(image_names)
        if use_cuda:
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
        pred_proba = model(inputs)

        actuals.extend(labels.tolist())
        all_img_names.extend(image_names)

        final_predicts = torch.argmax(pred_proba, dim=1)
        predictions.extend(pred_proba.tolist())
        total += len(labels)
        for a, b in zip(final_predicts, labels):
            if a == b:
                good += 1

    gathered_data["IMG"] = all_img_names
    gathered_data["LABEL"] = actuals
    gathered_data["PREDICTION"] = predictions

    df = pd.DataFrame(gathered_data)
    #         df.to_csv("/home/petigep/college/orak/digikep2/GOLD_TEST/great_cnn_pixels_76_res.csv", index=False)

    if eval_train:
        return good / total, df

    acc = good / total
    return acc, df

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MyImageFolder(ImageFolder):

    def __getitem__(self, index):
        # print(super(MyImageFolder, self).__getitem__(index))
        # print(self.imgs[index])

        return super(MyImageFolder, self).__getitem__(index) + (self.imgs[index][0],)

def load_data(data_path, tmp_path, args, shuffle=True):

    custom_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                           transforms.Resize((256, 256)),
                                           transforms.ToTensor(),
                                           ])

    dataset = MyImageFolder(root=tmp_path,
                            transform=custom_transform)

    # print(dataset.class_to_idx)

    return DataLoader(dataset=dataset,
                      batch_size=64,
                      shuffle=shuffle,
                      num_workers=4)


from PIL import Image
import torchvision.transforms.functional as TF

def get_image_tensor(path):

    image = Image.open(path)
    trns = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                        transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                        ])

    x = trns(image)
    x.unsqueeze_(0)
    return x