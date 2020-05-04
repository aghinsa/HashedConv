#ALEXNET
#source : https://github.com/icpm/pytorch-cifar10/blob/master/models/AlexNet.py

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from utils import evaluate,cifar10_loader

BATCH_SIZE = 1024
N_EPOCHS = 200
NUM_CLASSES = 10
LOAD_CKPT = None
CUDA = True

if CUDA :
	DEVICE = torch.device("cuda")
else:
	DEVICE = torch.device("cpu")

def load_alexnet():
	model = AlexNet()
	model.load_state_dict(torch.load("./checkpoint/model_30"))
	return model

from hasher import Hasher,get_conv_weights


class HashedConv(nn.Conv2d):

	def forward(self,x):
		new_weight,prob,bins,*_ = hasher(self.weight)
		# self.weight = nn.Parameter(new_weight)
		return self.conv2d_forward(x,new_weight)

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
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
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
        x = self.softmax(x)
        return x

def get_loss(labels,preds):
    ce = nn.NLLLoss(reduction="mean")
    l1 = ce(torch.log(preds),labels)
    return l1


if __name__ == "__main__":


    trainloader,testloader = cifar10_loader(512)

    CUDA =True

    model = AlexNet()
    LOAD_CKPT = 30
    if LOAD_CKPT is not None:
        LOAD_CKPT = f"./checkpoint/model_{LOAD_CKPT}"
        model.load_state_dict(torch.load(LOAD_CKPT))

    model.cuda()

    weights,biases,ip_sizes,names = get_conv_weights(model)
    hasher = Hasher(ip_sizes,16)
    hasher.cuda()

    params = list(model.parameters()) + list(hasher.parameters())
    optimizer = torch.optim.Adam(params)
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

            loss= get_loss(labels,preds)
    
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss/train', loss , global_step)

            global_step+=1
            print(f"loss:{loss}")

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