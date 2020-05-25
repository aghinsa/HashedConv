#ALEXNET
#source : https://github.com/icpm/pytorch-cifar10/blob/master/models/AlexNet.py

import torch
import numpy as np
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from alexnet import AlexNet as AlexNetNoHash
from utils import evaluate,cifar10_loader,get_weight_bins
from torch.distributions import Categorical

BATCH_SIZE = 2048
N_EPOCHS = 1000
NUM_CLASSES = 10
LOAD_CKPT = None
CUDA = True

TAU = .01

N_BINS = 8

class HashedConv(nn.Conv2d):
    def binary_init(self,size):
        w = np.random.choice([-1,1],size)
        return torch.from_numpy(w).float()

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.n_bins = N_BINS
        self.bins = nn.Parameter(torch.randn(self.n_bins,1))
        self.og_weight_size = self.weight.size()
        w_size = self.og_weight_size + (self.n_bins,)
        self.weight = nn.Parameter( self.binary_init(w_size) )

    def forward(self,x):
        # constructing probability matrix
        # prob = self.weight.clone().detach().view(-1,1) # [M,1]
        weight = torch.matmul(self.weight,torch.exp(self.bins)).squeeze()
        out = self.conv2d_forward(x,weight)
        return out



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


    def forward(self, x):
        x= self.conv1(x)
        x = self.ops1(x)

        x= self.conv2(x)
        x = self.ops2(x)

        x =  self.conv3(x)
        x =  self.ops3(x)

        x=  self.conv4(x)
        x =  self.ops4(x)

        x  =  self.conv5(x)
        x =  self.ops5(x)

        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x



def get_loss(labels,preds,model):
    ce = nn.CrossEntropyLoss(reduction="mean")
    l1 = ce(preds,labels)

    layers = [ "conv1","conv2","conv3","conv4","conv5" ]
    entropy_loss = 0
    for layer in layers:
        model_layer = getattr(model,layer)
        entropy_loss += Categorical(probs = model_layer.bins).entropy().sum()

    return l1 + entropy_loss



class WeightBinarizer(object):


    def __call__(self, module):
        # filter the variables to get the ones you want
        if type(module) == nn.Conv2d:
            w = module.weight.data
            w = torch.where(w>0,1,-1)
            module.weight = nn.Parameter(w)

if __name__ == "__main__":
    trainloader,testloader = cifar10_loader(batch_size=64,data_path="../data")
    CUDA =True

    model = AlexNet().cuda()
    layers = [ "conv1","conv2","conv3","conv4","conv5" ]


    optimizer = torch.optim.Adam(model.parameters())
    binarizer = WeightBinarizer()
    writer = SummaryWriter()

    global_step = 0

    for epoch in range(1,N_EPOCHS+1):

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()


            optimizer.zero_grad()

            preds = model(inputs)
            _,logits = torch.max(preds,1)

            loss= get_loss(labels,preds,model)
            loss.backward()
            optimizer.step()
            model.apply(binarizer)
            writer.add_scalar('loss/train', loss , global_step)

            global_step+=1
            print(f"loss:{loss}")


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