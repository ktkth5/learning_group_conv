
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from flgc import FLGC


class test_mnist(nn.Module):

    def __init__(self):
        super(test_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1,16,3)
        self.conv2 = FLGC(16,16,3,group_num=2)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 12 * 16, 128)
        self.dropout2 = nn.Dropout2d()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        if self.training:
            x = F.relu(self.conv1(x))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.dropout1(x)
            x = x.view(-1, 12 * 12 * 16)
            x = F.relu(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
        else:
            x = F.relu(self.conv1(x))
            x = self.pool(F.relu(self.conv2(x,if_flgc_is_next=False)))
            x = self.dropout1(x)
            x = x.view(-1, 12 * 12 * 16)
            x = F.relu(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
        return x

    def eval_set(self):
        self.index_list = []
        # self.index_list.append(self.conv1.before_inference())
        self.index_list.append(self.conv2.before_inference())





transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))])
trainset = torchvision.datasets.MNIST(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=32,
                                            shuffle=True,
                                            num_workers=1)

testset = torchvision.datasets.MNIST(root='./data',
                                        train=False,
                                        download=True,
                                        transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=32,
                                            shuffle=False,
                                            num_workers=1)

classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))




net = test_mnist()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

epochs = 1
for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 0:
            print(f"epoch[{epoch}:{i}/{len(trainloader)}] loss: {running_loss/100:.4f}")
            running_loss = 0.0

        if i>100:
            break

print('Finished Training')



correct = 0
total = 0


with torch.no_grad():
    for (images, labels) in testloader:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy: {:.2f} %%'.format(100 * float(correct/total)))

net.eval_set()
net.eval()
with torch.no_grad():
    for (images, labels) in testloader:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy: {:.2f} %%'.format(100 * float(correct/total)))