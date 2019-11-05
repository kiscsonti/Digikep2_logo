import torch.nn as nn
import torch.nn.functional as F


class LinearNet(nn.Module):
    def __init__(self, args):
        super(LinearNet, self).__init__()
        self.labels = args.number_of_labels
        self.batch_size = args.batch_size
        self.lls = args.img_size * args.img_size
        self.fc1 = nn.Linear(self.lls, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, self.labels)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):

        # print(x.shape)
        x = x.view(-1, self.lls)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        logits = F.log_softmax(x, dim=1)
        return logits


class CNN_64_64_32(nn.Module):
    def __init__(self, args):
        super(CNN_64_64_32, self).__init__()
        self.labels = args.number_of_labels
        self.batch_size = args.batch_size
        self.img_size = args.img_size
        self.h1_size = 64
        self.h2_size = 64
        self.h3_size = 48
        self.lls = self.h3_size*16*16
        self.conv1 = nn.Conv2d(9, self.h1_size, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.norm1 = nn.BatchNorm2d(self.h1_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(self.h1_size, self.h2_size, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.norm2 = nn.BatchNorm2d(self.h2_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(self.h2_size, self.h3_size, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.norm3 = nn.BatchNorm2d(self.h3_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc1 = nn.Linear(self.lls, 512)
        self.fc2 = nn.Linear(512, self.labels)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        batch = x.shape[0]
        x = x.view(batch, 9, (int(self.img_size/2)-1) * 2, (int(self.img_size/2)-1) * 2)
        # print(x.shape)
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        # print(x.shape)
        x = self.pool(F.relu(self.norm2(self.conv2(x))))
        # print(x.shape)
        x = self.pool(F.relu(self.norm3(self.conv3(x))))
        # print(x.shape)
        x = x.view(-1, self.lls)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        logits = F.log_softmax(x, dim=1)
        return logits


class CNN_32_64_32(nn.Module):
    def __init__(self, args):
        super(CNN_32_64_32, self).__init__()
        self.labels = args.number_of_labels
        self.batch_size = args.batch_size
        self.img_size = args.img_size
        self.h1_size = 32
        self.h2_size = 64
        self.h3_size = 48
        self.lls = self.h3_size*8*8
        self.conv1 = nn.Conv2d(9, self.h1_size, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.norm1 = nn.BatchNorm2d(self.h1_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(self.h1_size, self.h2_size, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.norm2 = nn.BatchNorm2d(self.h2_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(self.h2_size, self.h3_size, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.norm3 = nn.BatchNorm2d(self.h3_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc1 = nn.Linear(self.lls, 512)
        self.fc2 = nn.Linear(512, self.labels)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        batch = x.shape[0]
        x = x.view(batch, 9, (int(self.img_size/4)-1) * 2, (int(self.img_size/4)-1) * 2)
        # print(x.shape)
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        # print(x.shape)
        x = self.pool(F.relu(self.norm2(self.conv2(x))))
        # print(x.shape)
        x = self.pool(F.relu(self.norm3(self.conv3(x))))
        # print(x.shape)
        x = x.view(-1, self.lls)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        logits = F.log_softmax(x, dim=1)
        return logits


class BasicNet_14(nn.Module):
    def __init__(self, args):
        super(BasicNet_14, self).__init__()
        self.labels = args.number_of_labels
        self.batch_size = args.batch_size
        self.h1_size = 32
        self.h2_size = 32
        self.lls = self.h2_size*32*32
        self.conv1 = nn.Conv2d(1, self.h1_size, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.norm1 = nn.BatchNorm2d(self.h1_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(self.h1_size, self.h2_size, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.norm2 = nn.BatchNorm2d(self.h2_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc1 = nn.Linear(self.lls, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, self.labels)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        # print(x.shape)
        x = self.pool(F.relu(self.norm2(self.conv2(x))))
        # print(x.shape)
        x = x.view(-1, self.lls)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        logits = F.log_softmax(x, dim=1)
        return logits


class V2_CNN(nn.Module):
    def __init__(self, labels, args):
        super(V2_CNN, self).__init__()
        self.labels = labels
        self.batch_size = args.batch_size
        self.h1_size = 32
        self.h2_size = 64
        self.h3_size = 64
        self.lls = self.h3_size*14*14
        self.conv1 = nn.Conv2d(3, self.h1_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.norm1 = nn.BatchNorm2d(self.h1_size)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=1, dilation=1, ceil_mode=False)

        self.conv2 = nn.Conv2d(self.h1_size, self.h2_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.norm2 = nn.BatchNorm2d(self.h2_size)

        self.conv3 = nn.Conv2d(self.h2_size, self.h3_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.norm3 = nn.BatchNorm2d(self.h3_size)
        self.dropout = nn.Dropout(p=0.25)


        self.fc1 = nn.Linear(self.lls, 256)
        self.fc2 = nn.Linear(256, self.labels)


    def forward(self, x):
        # print(x.shape)
        x = self.pool(self.norm1(F.relu(self.conv1(x))))
        # print(x.shape)
        x = self.pool(self.norm2(F.relu(self.conv2(x))))

        x = self.dropout(self.pool(self.norm3(F.relu(self.conv3(x)))))
        # print(x.shape)
        x = x.view(-1, self.lls)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class LinearNet_old(nn.Module):
    def __init__(self, labels, args):
        super(LinearNet, self).__init__()
        self.labels = labels
        self.batch_size = args.batch_size
        self.lls = 224*224*3
        self.fc1 = nn.Linear(self.lls, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, self.labels)

    def forward(self, x):
        x = x.view(-1, self.lls)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BasicNet_014(nn.Module):
    def __init__(self, labels, args):
        super(BasicNet_014, self).__init__()
        self.labels = labels
        self.batch_size = args.batch_size
        self.h1_size = 128
        self.h2_size = 64
        self.h3_size = 64
        self.lls_1 = self.h2_size*28*28
        self.lls = self.h3_size*14*14
        self.conv1 = nn.Conv2d(3, self.h1_size, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.norm1 = nn.BatchNorm2d(self.h1_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(self.h1_size, self.h2_size, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.norm2 = nn.BatchNorm2d(self.h2_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(self.h2_size, self.h3_size, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.norm3 = nn.BatchNorm2d(self.h3_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc1 = nn.Linear(self.lls, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, self.labels)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        # print(x.shape)
        x = self.pool(F.relu(self.norm2(self.conv2(x))))
        x = self.pool(F.relu(self.norm3(self.conv3(x))))
        # print(x.shape)
        x = x.view(-1, self.lls)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        logits = F.log_softmax(x, dim=1)
        return logits


class BasicNet_007(nn.Module):
    def __init__(self, labels, args):
        super(BasicNet_007, self).__init__()
        self.labels = labels
        self.batch_size = args.batch_size
        self.fc1 = nn.Linear(224*224*3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, self.labels)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # print(x.shape)
        x = x.view(-1, 224*224*3)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        logits = F.log_softmax(x, dim=1)
        return logits
