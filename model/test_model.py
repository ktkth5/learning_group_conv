import torch
import torch.nn as nn
import torch.nn.functional as F

from flgc import FLGC


class test_cifar(nn.Module):

    def __init__(self):
        super(test_cifar, self).__init__()
        self.conv1 = FLGC(3,6,5,group_num=2)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = FLGC(6,16,5,group_num=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        if self.training:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            x = self.pool(F.relu(self.conv1(x,if_flgc_is_next=True)))
            x = self.pool(F.relu(self.conv2(x,self.index_list[0],if_flgc_is_next=False)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        return x

    def eval_set(self):
        self.index_list = []
        self.index_list.append(self.conv1.before_inference())
        self.index_list.append(self.conv2.before_inference())

