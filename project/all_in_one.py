import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
from PIL import Image
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from models import BasicNet


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
        return img, label

    def __len__(self):
        return self.y.shape[0]


batch_size = 64

custom_transform2 = transforms.Compose([transforms.Grayscale(),
                                       transforms.ToTensor()])
custom_transform = transforms.Compose([transforms.Resize((224,224)),
                                       transforms.ToTensor(),
                                       ])


train_dataset = LogoDataset(txt_path='data/train_data_merged.csv',
                              img_dir='data/flickr_logos_27_dataset/flickr_logos_27_dataset_images',
                              transform=custom_transform)

trainloader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4)

test_dataset = LogoDataset(txt_path='data/test_data_merged.csv',
                              img_dir='data/flickr_logos_27_dataset/flickr_logos_27_dataset_images',
                              transform=custom_transform)

testloader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4)


labels_count = set()
for item in test_dataset.y:
    labels_count.add(item)
print('Number of labels:', len(labels_count))
# print(labels_count)
# sys.exit(0)

device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")
# model = models.densenet121(pretrained=True)
# model = models.densenet121(pretrained=True)
model = BasicNet(len(labels_count), batch_size)
print(model)


model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Trainable parameters: ", params)

# for param in model.parameters():
#     param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 512),
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(512, len(labels_count)),
                         nn.Softmax(dim=1))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)
model.to(device)

epochs = 100
running_loss = 0
train_losses, test_losses = [], []
for epoch in range(epochs):
    for inputs, labels in trainloader:
        inputs = inputs.to(device)
        #         print(labels)
        labels = labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    train_losses.append(running_loss / len(trainloader))
    test_losses.append(test_loss / len(testloader))
    print(f"Epoch {epoch + 1}/{epochs}.. "
          f"Train loss: {running_loss / (epoch+1):.5f}.. "
          f"Test loss: {test_loss / len(testloader):.5f}.. "
          f"Test accuracy: {accuracy / len(testloader):.5f}")
    running_loss = 0
    model.train()
torch.save(model, 'test_model_1.pth')