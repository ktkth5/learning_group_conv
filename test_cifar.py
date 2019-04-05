
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
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

    epochs = 20
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
            if i % 200 == 0:
                print(f"epoch[{epoch}:{i}/{len(trainloader)}] loss: {running_loss/200:.4f}")
                running_loss = 0.0

            # if i>300:
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

    # a = np.array([0,1,2,3,4,0,1,2,3,4])
    # print(np.where(a==0))

    # for debug
    # net = model.MobileNetV2_flgc(n_class=10)
    # net = loss_function.add_flgc_loss(net)
    # net = model.model_module.add_eval_set(net)
    # net.eval_set()