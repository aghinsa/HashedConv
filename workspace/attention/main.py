import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from alexnet import load_alexnet
from utils import evaluate,cifar10_loader
from hasher import Hasher,set_conv_weights,get_conv_weights,hash_loss


import torch.optim as optim

BATCH_SIZE = 1024
N_EPOCHS = 50

CUDA =True

LOAD_CKPT = None

if __name__ == "__main__":

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    trainloader,testloader = cifar10_loader(batch_size=128)

    model = load_alexnet().cuda()
    weights,biases,ip_sizes,names = get_conv_weights(model)
    hasher = Hasher(ip_sizes,16)
    hasher.cuda()

    optimizer = torch.optim.Adam(hasher.parameters())
    writer = SummaryWriter()

    global_step = 0
    ce = nn.NLLLoss(reduction="mean")

    for epoch in range(1,N_EPOCHS+1):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            loss = 0

            weights,biases,_,_ = get_conv_weights(model)
            new_weights=[]
            for weight in chain(*[weights]):
                new_weight,x_up,P,Q,prob = hasher(weight)
                new_weights.append(new_weight)
                loss += hash_loss(weight,new_weight,x_up,P,Q,prob)

            set_conv_weights(model,new_weights,[],names)

            preds = model(inputs)
            loss += ce(torch.log(preds),labels)

            print(f"loss:{loss}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch : {epoch} complete")
        if epoch%10 == 0:
            hasher.eval()
            weights,biases,_,_ = get_conv_weights(model)
            new_weights=[]
            for weight in chain(*[weights]):
                new_weight,x_up,P,Q,prob = hasher(weight)
                new_weights.append(new_weight)

            test_acc = evaluate(model,testloader,cuda=True)
            train_acc = evaluate(model,trainloader,cuda=True)

            print(f"Epoch {epoch} complete:")
            print(f"\ttest Accuracy : {test_acc}")
            print(f"\ttrain Accuracy : {train_acc}")

            hasher.train()

    print('Finished Training')