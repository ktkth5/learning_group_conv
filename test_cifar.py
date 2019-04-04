
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from flgc import FLGC
import loss_function
import model
import utils


class _test_cifar(nn.Module):

    def __init__(self):
        super(_test_cifar, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
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
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        return x

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
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        return x

    def eval_set(self):
        for module in self.children():
            if isinstance(module, FLGC):
                module.before_inference()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = test_cifar()
    # net = model.MobileNetV2(n_class=10)
    net = model.MobileNetV2_flgc(n_class=10)
    # net = loss_function.add_flgc_loss(net)
    net = model.model_module.add_eval_set(net)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    epochs = 5
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels) #+ net.flgc_loss()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:
                print(f"epoch[{epoch}:{i}/{len(trainloader)}] loss: {running_loss/100:.4f}")
                running_loss = 0.0

            # if i>500:
            #     break

    print('Finished Training')



    correct = 0
    total = 0

    import time

    net.eval()
    with torch.no_grad():
        start = time.time()
        for (images, labels) in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().cpu().item()
    print('Accuracy: {:.2f} %%'.format(100 * float(correct/total)), f"Time: {time.time()-start:.2f}")

    # class_correct = list(0. for i in range(10))
    # class_total = list(0. for i in range(10))
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs, 1)
    #         c = (predicted == labels).squeeze()
    #         for i in range(4):
    #             label = labels[i]
    #             class_correct[label] += c[i].item()
    #             class_total[label] += 1
    #
    # for i in range(10):
    #     print('Accuracy of %5s : %2d %%' % (
    #         classes[i], 100 * class_correct[i] / class_total[i]))


    net.eval_set()
    with torch.no_grad():
        start = time.time()
        for (images, labels) in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().cpu().item()
    print('Accuracy: {:.2f} %%'.format(100 * float(correct/total)), f"Time: {time.time()-start:.2f}")

    # class_correct = list(0. for i in range(10))
    # class_total = list(0. for i in range(10))
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs, 1)
    #         c = (predicted == labels).squeeze()
    #         for i in range(4):
    #             label = labels[i]
    #             class_correct[label] += c[i].item()
    #             class_total[label] += 1
    #
    #
    # for i in range(10):
    #     print('Accuracy of %5s : %2d %%' % (
    #         classes[i], 100 * class_correct[i] / class_total[i]))


    # if calc flop
    # from utils.benchmark import add_flops_counting_methods
    # bs = 2
    # img_sizes = [32]
    # for img_size in img_sizes:
    #     m = add_flops_counting_methods(net)
    #     m.start_flops_count()
    #     batch = torch.FloatTensor(bs, 3, img_size, img_size)
    #     _ = m(batch)
    #     print("normal conv :", f"img size: {img_size} flops: {m.compute_average_flops_cost() / 1e9 / 2}")
    #
    # for img_size in img_sizes:
    #     m = _test_cifar()
    #     m = add_flops_counting_methods(m)
    #     m.start_flops_count()
    #     batch = torch.FloatTensor(bs, 3, img_size, img_size)
    #     _ = m(batch)
    #     print("flgc conv   :", f"img size: {img_size} flops: {m.compute_average_flops_cost() / 1e9 / 2}")

if __name__=="__main__":
    main()