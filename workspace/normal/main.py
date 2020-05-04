import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
from  itertools import chain

from blocks import ProjectDown,ProjectUp
from torch.utils.tensorboard import SummaryWriter
from hashnet import HashNet,CentroidNet
from utils import evaluate,cifar10_loader

from alexnet import load_alexnet
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision

from hasher import Hasher,set_conv_weights,get_conv_weights
from utils import cifar10_loader,evaluate

class HashedConv(nn.Module):
    def custom_init(self,hasher,conv):
        self.conv = conv
        self.hasher = hasher
    def forward(self,x):
        if not self.training:
            return self.conv(x)

        new_weight,prob,bins = self.hasher(torch.tensor(self.conv.weight))
        self.conv.weight = nn.Parameter(new_weight)
        self.hasher.set_extras(
            {
                "bins":bins,
                "prob":prob
            }
        )
        return self.conv(x)

def to_hashed(model,hasher):
    hasher.pretrained = True
    for k,v in model._modules.items():
        if "conv" in k:
            conv = HashedConv()
            conv.custom_init(hasher,v)
            model._modules[k]=conv
    return model

def get_loss(labels,preds):
    ce = nn.NLLLoss(reduction="mean")
    l1 = ce(torch.log(preds),labels)
    return l1
if __name__ == "__main__":
    model = load_alexnet().cuda()
    trainloader,testloader = cifar10_loader()
    klloss = nn.KLDivLoss(reduction="sum")
    hasher = Hasher(16)
    hasher.cuda()
    # hasher.load_state_dict(torch.load("./checkpoint/pretrained_hasher_no"))
    hasher.pretrained = True

    model = to_hashed(model,hasher)

    optimizer = torch.optim.Adam(model.parameters())
    writer = SummaryWriter()

    global_step = 0

    for epoch in range(1,500):

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            preds= model(inputs)
            _,logits = torch.max(preds,1)
            bins  = hasher.extras["bins"]
            prob  = hasher.extras["prob"]

            print(f" bins : {bins}")
            loss= get_loss(labels,preds)
            # loss += klloss(P.log(),prob)

            loss.backward()
            optimizer.step()


            global_step+=1
            print(f"loss:{loss}")


        if epoch%10==0:
            torch.save(model.state_dict(), f"./checkpoint/model1_{epoch}")


            model.eval()
            test_acc = evaluate(model,testloader,cuda = True)
            train_acc = evaluate(model,trainloader,cuda = True)

            print(f"Epoch {epoch} complete:")
            print(f"\ttest Accuracy : {test_acc}")
            print(f"\ttrain Accuracy : {train_acc}")
            model.train()