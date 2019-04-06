import time
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256*2,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # net = model.MobileNetV2(n_class=10)
    net = model.MobileNetV2_flgc(n_class=10)
    # net = loss_function.add_flgc_loss(net)
    net = model.model_module.add_eval_set(net)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)

    print("Start Training")
    epochs = 1
    for epoch in range(epochs):
        scheduler.step()
        train_loss = train(net, trainloader, criterion, optimizer, epoch)
        val_acc = validation(net, testloader, criterion, epoch)
        print(f"Epoch: [{epoch}/{epochs}]"
              f"Train Loss: {train_loss:.4f}\t"
              f"Val Acc: {val_acc:.2f} %%")

            # if i>300:
            #     break

    print('Finished Training')

    final_val_acc, class_correct, class_total = final_validation(net, testloader)
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
              classes[i], 100 * class_correct[i] / class_total[i]))
    print(f"Final Accuracy after reordering: {final_val_acc:.2f} %%")

    # correct = 0
    # total = 0
    #
    # import time
    #
    # net.eval()
    # with torch.no_grad():
    #     start = time.time()
    #     for (images, labels) in testloader:
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().cpu().item()
    # print('Accuracy: {:.2f} %%'.format(100 * float(correct/total)), f"Time: {time.time()-start:.2f}")

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


    # net.eval_set()
    # with torch.no_grad():
    #     start = time.time()
    #     for (images, labels) in testloader:
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().cpu().item()
    # print('Accuracy: {:.2f} %%'.format(100 * float(correct/total)), f"Time: {time.time()-start:.2f}")

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


def train(model, train_loader, criterion, optimizer, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    running_loss = 0.0
    loss_avg = AverageMeter()
    model.train()
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)  # + net.flgc_loss()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        loss_avg.update(loss.item(), inputs.size(0))
        # if i % 200 == 0:
        #     print(f"epoch[{epoch}:{i}/{len(train_loader)}] loss: {running_loss/200:.4f}")
        #     running_loss = 0.0
    return loss_avg.avg

def validation(model, val_loader, criterion, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval()
    with torch.no_grad():
        start = time.time()
        for (images, labels) in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().cpu().item()
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    # print('Accuracy: {:.2f} %%'.format(100 * float(correct / total)), f"Time: {time.time()-start:.2f}")
    return 100 * float(correct / total)


def final_validation(model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval_set()
    with torch.no_grad():
        start = time.time()
        for (images, labels) in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().cpu().item()
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    # print('Accuracy: {:.2f} %%'.format(100 * float(correct / total)), f"Time: {time.time()-start:.2f}")

    return 100 * float(correct / total), class_correct, class_total


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__=="__main__":
    main()

    # a = np.array([0,1,2,3,4,0,1,2,3,4])
    # print(np.where(a==0))

    # for debug
    # net = model.MobileNetV2_flgc(n_class=10)
    # net = loss_function.add_flgc_loss(net)
    # net = model.model_module.add_eval_set(net)
    # net.eval_set()

    # n = np.arange(480).reshape((5, 4, 4, 6))
    # a = torch.from_numpy(n)
    # perm = torch.LongTensor([0, 2, 1, 3])
    # print(a[[0,1,2]][:,[0,1]].shape)
    # print(a[[1,2,0]].shape)
    # print(a[:,[0,1,2]].shape)
    # print(a[:,[1,2,0]].shape)

    # print(a[:, perm][:,perm])

    # a = nn.Conv2d(1,3,kernel_size=2)
    # a.weight.data = torch.zeros(3,1,2,2)
