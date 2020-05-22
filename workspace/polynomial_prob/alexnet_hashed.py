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



class HashedConv(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.n_bins = 16
        self.bins = nn.Parameter(torch.randn(self.n_bins))
        self.weight = None
        self.centroids = None # keeps initial bins

        # Have n_bins number of polynomial each with n_bins + 2 coefficients
        # the Pi(X) outputs a similarity measure of how close X is to bin i
        # for selecting bins of X use softmax( [Pi(X) | i in n_bins] )
        self.polynomial_weights = nn.Parameter( torch.randn(self.n_bins,self.n_bins + 1) )
        self.polynomial_biases = nn.Parameter( torch.randn(self.n_bins,1) )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        # constructing probability matrix
        # prob = self.weight.clone().detach().view(-1,1) # [M,1]
        prob = self.weight.detach().view(-1,1) # [M,1]

        prob = torch.cat(
            [
                prob,
                self.bins.view(1,-1).repeat( prob.size()[0],1 )
            ],
            1
        ) # [M,n_bins +1 ]

        prob = torch.matmul(prob, self.polynomial_weights.t()) # [M,n_bins]
        prob = prob + self.polynomial_biases.t()
        prob = prob - self.bins.view(1,-1)
        prob = 1/1+torch.pow(prob,2)

        # gumbel-softmax requires inputs to be unnormalized log probablities
        # so no need to take softmax over prob
        prob = F.gumbel_softmax(prob,tau = 1 , hard = True) # [M,n_dim]
        # prob = self.softmax(prob) # [M,n_dim]
        self.prob = prob
        new_weight = (prob*self.bins).sum(dim=-1).view(self.weight.size() )
        self.new_weight = new_weight
        out = self.conv2d_forward(x,new_weight)
        return out




class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = HashedConv(3, 64, kernel_size=3, stride=2, padding=1,)
        self.ops1 = nn.Sequential(
                        nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = HashedConv(64, 192, kernel_size=3, padding=1,)
        self.ops2 = nn.Sequential(
                        nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )
        self.bn2 = nn.BatchNorm2d(192)

        self.conv3 = HashedConv(192, 384, kernel_size=3, padding=1,)
        self.ops3 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(384)

        self.conv4 = HashedConv(384, 256, kernel_size=3, padding=1,)
        self.ops4 = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(256)

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

    def custom_init(self,pretrained_model,n_bins):
        layers = [ "conv1","conv2","conv3","conv4","conv5" ]

        for layer in layers:
            pretrained = getattr(pretrained_model,layer)
            curr = getattr(self,layer)

            trained_weight = pretrained.weight.clone().detach()
            bins = get_weight_bins(trained_weight.cpu(), n_bins)
            curr.bins = nn.Parameter(torch.from_numpy(bins).cuda())
            curr.centroids = torch.from_numpy(bins).cuda()
            curr.weight = nn.Parameter(trained_weight)

    def forward(self, x):
        x= self.conv1(x)
        x = self.ops1(x)
        x = self.bn1(x)

        x= self.conv2(x)
        x = self.ops2(x)
        x = self.bn2(x)

        x =  self.conv3(x)
        x =  self.ops3(x)
        x = self.bn3(x)

        x=  self.conv4(x)
        x =  self.ops4(x)
        x = self.bn4(x)

        x  =  self.conv5(x)
        x =  self.ops5(x)

        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

def get_loss(labels,preds):
    ce = nn.CrossEntropyLoss(reduction="mean")
    l1 = ce(preds,labels)
    layers = [ "conv1","conv2","conv3","conv4","conv5" ]

    return l1

def get_model_hashed_weight_loss(model,pretrained_model):
    mse_fn = nn.MSELoss(reduce="mean")

    entropy_loss = 0
    mse_loss = 0

    # mse loss for each weight with the already hashed weights
    # entropy loss for prob in each layer
    layers = [ "conv1","conv2","conv3","conv4","conv5" ]
    for layer in layers:
        model_layer = getattr(model,layer)
        pretrained_layer = getattr(pretrained_model ,layer)
        entropy_loss += Categorical(probs = model_layer.prob).entropy().sum()

        mse_loss +=(
            mse_fn(model_layer.new_weight,pretrained_layer.weight) +
            mse_fn(model_layer.bins,model_layer.centroids.cuda())
        )
    return entropy_loss + mse_loss



if __name__ == "__main__":
    trainloader,testloader = cifar10_loader(batch_size=128,data_path="../data")

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    CUDA =True

    model = AlexNet()
    pretrained_model = AlexNetNoHash()
    pretrained_model.load_state_dict(torch.load("./alexnet_pretrained"))
    model.custom_init(pretrained_model,n_bins=16)

    if CUDA:
        model.cuda()
        pretrained_model.cuda()

    optimizer = torch.optim.Adam(model.parameters())
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

            # loss= get_loss(labels,preds) + get_model_hashed_weight_loss(model,pretrained_model)
            loss=  get_model_hashed_weight_loss(model,pretrained_model)

            loss.backward()
            optimizer.step()

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