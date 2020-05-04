import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from alexnet import AlexNet

import torch.optim as optim

BATCH_SIZE = 1024
N_EPOCHS = 50

import torchvision.models as models

LOAD_CKPT = None

def evaluate(model,loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()
            p = model(inputs)
            p = torch.max(p.data, 1)

            total += labels.size(0)

            correct += (p == labels).sum().item()

    return correct[0]/total

def hash_loss(labels,logits,logits_hash,ps,qs,probs):
    kl = nn.KLDivLoss(reduction="sum")
    ce = nn.NLLLoss(reduction="mean")


    l1 = ce(torch.log(logits),labels)
    l2 = ce(torch.log(logits_hash),labels)

    kl1 = 0

    # kl expects target as prob and inputs as log prob
    for i in range(len(ps)):
        P = ps[i]
        Q = qs[i]
        Q = torch.log(Q)
        kl1 += kl(Q,P)




    return l1,l2,kl1



def get_accuracy(labels,logits,reduce_mean):
    acc = (labels == logits).sum()
    if reduce_mean:
        acc = acc/logits.size()[0]
    return acc
if __name__ == "__main__":
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    download_data = False
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=download_data,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=download_data, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



    CUDA =True

    model = models.alexnet(pretrained=True)

    if LOAD_CKPT is not None:
        LOAD_CKPT = f"./checkpoint/model_{LOAD_CKPT}"
        model.load_state_dict(torch.load(LOAD_CKPT))

    if CUDA:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters())
    writer = SummaryWriter()

    global_step = 0


    model.eval()
    test_acc,test_acc_h = evaluate(model,testloader)
    train_acc,train_acc_h = evaluate(model,trainloader)

    print(f"Epoch {epoch} complete:")
    print(f"\ttest Accuracy : {test_acc}")
    print(f"\ttest Accuracy_hash : {test_acc_h}")
    print(f"\ttrain Accuracy : {train_acc}")
    print(f"\ttrain Accuracy_hash : {train_acc_h}")
    model.train()

    print('Finished Training')