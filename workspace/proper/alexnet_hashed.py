#ALEXNET
#source : https://github.com/icpm/pytorch-cifar10/blob/master/models/AlexNet.py

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
from utils import evaluate,cifar10_loader

BATCH_SIZE = 2048
N_EPOCHS = 1000
NUM_CLASSES = 10
LOAD_CKPT = None
CUDA = True

TAU = .01



class HashedConv(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.n_bins = 16
        self.bins = nn.Parameter(torch.randn(self.n_bins))
        self.probs = nn.Parameter(torch.randn(self.weight.size() + (self.n_bins,) ))
        self.i_out = None
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,i):
        # probs = F.gumbel_softmax(self.probs,tau=TAU,hard=True,dim=-1)
        probs = self.softmax(self.probs/TAU)
        new_weight = (probs*self.bins).sum(dim=-1)
        self.i_out = self.conv2d_forward(i,new_weight)
        return self.i_out




class Entropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x is unnormalized weight
        x = nn.Softmax(dim=-1)(x)
        b = x * torch.log(x)
        b = -1.0 * b.mean(dim=-1)
        b = b.mean()
        return b

class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = HashedConv(3, 64, kernel_size=3, stride=2, padding=1,)
        self.ops1 = nn.Sequential(
                        nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )

        self.conv2 = HashedConv(64, 192, kernel_size=3, padding=1,)
        self.ops2 = nn.Sequential(
                        nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )

        self.conv3 = HashedConv(192, 384, kernel_size=3, padding=1,)

        self.ops3 = nn.ReLU(inplace=True)

        self.conv4 = HashedConv(384, 256, kernel_size=3, padding=1,)

        self.ops4 = nn.ReLU(inplace=True)

        self.conv5 = HashedConv(256, 256, kernel_size=3, padding=1,)
        self.ops5 = nn.Sequential(
                        nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )
        self.softmax = nn.Softmax(dim=1)

        self.hashed_loss = None
        self.entropy_loss = None
        self.mse = nn.MSELoss()
        self.entropy = Entropy()

        self.rx =None
        self.ix =None


            # div_loss = kldiv_loss_fn(r_preds,i_preds)
    def set_hashed_loss(self):

        self.entropy_loss = (
            self.entropy(self.conv1.probs) +
            self.entropy(self.conv2.probs) +
            self.entropy(self.conv3.probs) +
            self.entropy(self.conv4.probs) +
            self.entropy(self.conv5.probs) +
            self.entropy(self.conv1.bins.view(1,-1)) +
            self.entropy(self.conv2.bins.view(1,-1)) +
            self.entropy(self.conv3.bins.view(1,-1)) +
            self.entropy(self.conv4.bins.view(1,-1)) +
            self.entropy(self.conv5.bins.view(1,-1))
            )



    def forward(self, x):
        ix= self.conv1(x)
        ix = self.ops1(ix)

        ix= self.conv2(ix)
        ix = self.ops2(ix)

        ix =  self.conv3(ix)
        ix =  self.ops3(ix)

        ix=  self.conv4(ix)
        ix =  self.ops4(ix)

        ix  =  self.conv5(ix)
        ix =  self.ops5(ix)
        
        ix = ix.view(ix.size(0), 256 * 2 * 2)
        ix = self.classifier(ix)

        # if self.training:
        #     self.set_hashed_loss()
        return ix


def kldiv_loss_fn(p,q):

    l = nn.LogSoftmax(dim=-1)
    x = l(p) -l(q)
    x = nn.Softmax(dim=-1)(p)* x
    x = x.sum()
    return x

if __name__ == "__main__":



    trainloader,testloader = cifar10_loader(BATCH_SIZE)

    CUDA =True

    model = AlexNet()

    model.cuda()
    # model.load_state_dict(torch.load("./checkpoint/model_20"))

    ce_loss_fn = nn.CrossEntropyLoss()
    # kldiv_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(model.parameters())
    writer = SummaryWriter()
    softmax_fn = nn.Softmax(dim=-1)
    global_step = 0

    for epoch in range(1,N_EPOCHS+1):

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()


            optimizer.zero_grad()

            preds = model(inputs)


            class_loss = ce_loss_fn(preds,labels)
            # e_loss = model.entropy_loss


            loss = class_loss 

            print(f"class_loss : {class_loss}")
            # print(f"entropy_loss : {e_loss}")



            loss.backward()
            optimizer.step()

            writer.add_scalar('loss/classification_loss', class_loss , global_step)
            # writer.add_scalar('loss/entropy_loss', e_loss , global_step)
            writer.add_scalar('loss/train', loss , global_step)

            global_step+=1
            print(f"{epoch}:loss:{loss}")

        print(f"Epoch : {epoch} done")
        if epoch%10==0:
            torch.save(model.state_dict(), f"./checkpoint/model_{epoch}")


            model.eval()
            test_acc = evaluate(model,testloader,cuda = CUDA)
            train_acc = evaluate(model,trainloader,cuda = CUDA)

            print(f"Epoch {epoch} complete:")
            print(f"\ttest Accuracy : {test_acc}")
            print(f"\ttrain Accuracy : {train_acc}")
            model.train()

    print('Finished Training')