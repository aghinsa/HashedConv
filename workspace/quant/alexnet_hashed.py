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
from utils import evaluate,cifar10_loader
from torch.distributions import Categorical

from quantizer import HashedConv

BATCH_SIZE = 2048
N_EPOCHS = 1000
NUM_CLASSES = 10
LOAD_CKPT = None
CUDA = True

TAU = .01

def init_by_power(n):
    l = []
    p = 1
    for i in range(n):
        l.append(p)
        p*=2
    l = np.array(l).reshape(-1,1)
    l = torch.from_numpy(l).float()
    return l




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

    def get_hash_loss(self):
        return self.conv1.hash_loss + self.conv2.hash_loss + self.conv3.hash_loss + self.conv4.hash_loss + self.conv5.hash_loss



def get_loss(labels,preds,model):
    ce = nn.CrossEntropyLoss(reduction="mean")
    l1 = ce(preds,labels)

    return l1



if __name__ == "__main__":
    trainloader,testloader = cifar10_loader(batch_size=128,data_path="../data")

    model = AlexNet()
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters())
    writer = SummaryWriter()

    global_step = 0

    for epoch in range(1,N_EPOCHS+1):

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            inputs = inputs.cuda()
            labels = labels.cuda()


            optimizer.zero_grad()

            preds = model(inputs)
            _,logits = torch.max(preds,1)

            loss= get_loss(labels,preds,model) + model.get_hash_loss()
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss/train', loss , global_step)

            global_step+=1
            print(f"loss:{loss}")


        if epoch%1==0:
            # torch.save(model.state_dict(), f"./checkpoint/model_{epoch}")
            trl,tsl = cifar10_loader(batch_size=4096,data_path="../data")
            test_acc = evaluate(model,tsl,cuda = True)
            train_acc = evaluate(model,trl,cuda = True)
            print(f"Epoch {epoch} complete:")
            print(f"\ttest Accuracy : {test_acc}")
            print(f"\ttrain Accuracy : {train_acc}")

            model.eval()
            test_acc = evaluate(model,tsl,cuda = True)
            train_acc = evaluate(model,trl,cuda = True)
            print(f"\teval_test Accuracy : {test_acc}")
            print(f"\teval_train Accuracy : {train_acc}")
            model.train()

        if epoch%5 == 0:
            torch.save(model.state_dict(), f"./checkpoint/model_{epoch}")


    print('Finished Training')