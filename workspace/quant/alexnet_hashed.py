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
from sklearn.linear_model import SGDRegressor

from quant_utils import get_binary_encodings

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

class HashedConv(nn.Conv2d):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.n_bins = 4
        self.n_fs = 16
        # self.w.size (outc,inc,k,k)
        self.n_out = self.weight.size()[0]
        self.n_fs = min(self.n_fs,self.n_out)
        self.f_bins = torch.rand(self.n_fs,self.n_bins,).cuda()


        self.all_encodings = np.array(get_binary_encodings(self.n_bins))
        self.all_encodings = torch.from_numpy(self.all_encodings) # [2^n_bins,n_bins]
        self.all_encodings = self.all_encodings.float().cuda()

        self.sgd = SGDRegressor(max_iter=100, tol=1e-3,warm_start=True)
        self.selected_encoding = None

    def forward(self,x):

        # Calculating function coefficients
        with torch.no_grad():
            w_master = self.weight.clone().detach()

            # Optimizing hashed weights
            for _ in range(2):
                channel_step = self.out_channels//self.n_fs
                new_fs = []
                new_ws = []

                for fidx in range(0,self.n_fs):
                    # select hash functions for the channel batch
                    f = self.f_bins[fidx].reshape(-1,1) #[nb,1]

                    if fidx != self.n_fs-1:
                        wx = w_master[fidx*channel_step:(fidx+1)*channel_step,:,:,: ] # [channel_step,inc,k,k]
                    else:
                        wx = w_master[fidx*channel_step:,:,:,: ] # [channel_step,inc,k,k]

                    # finding encoding for qhich quant level is closest to w
                    w = wx.reshape(-1,1)
                    quant_levels = torch.matmul(self.all_encodings, f) # [2^nb,1]

                    w = torch.abs( w - quant_levels.t()) # [m,2^nb]
                    
                    idx = torch.argmin(w,dim = -1)

                    selected_encoding = self.all_encodings[idx] #[m,nbins]

                    # From the selected encodings generate hash values
                    new_w = torch.matmul(selected_encoding,f).reshape(wx.size())
                    new_ws.append(new_w)

                    # optimizing coefficients which minimizes square loss with
                    # current unhashed weights
                    if self.training:
                        self.sgd.fit(selected_encoding.cpu(),wx.reshape(-1).cpu())
                        f_new_bin = self.sgd.coef_
                        f_new_bin = torch.from_numpy(f_new_bin).reshape(1,-1).float().cuda()
                        new_fs.append(f_new_bin)

                        # s = selected_encoding #[m,nb]

                        # st = s.t() #[nb,m]
                        # new_bins = torch.matmul(st,s) #[nb,nb]
                        # new_bins = torch.inverse(new_bins)
                        # new_bins = torch.matmul(new_bins,st) # [nb,m]
                        # new_bins = torch.matmul(new_bins,wx.reshape(-1,1)) #[nb,1]

                weight_hashed = torch.cat(new_ws,axis=0)
                self.weight_hashed = weight_hashed

                if self.training:
                    new_fs = torch.cat(new_fs,axis=0)
                    alpha = 0.9
                    self.f_bins = (1-alpha)*self.f_bins + alpha*(new_fs)


        if self.training:
            r = self.conv2d_forward(x,self.weight)
            return r
        else:
            i = self.conv2d_forward(x,self.weight_hashed)
            return i


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

    return l1



if __name__ == "__main__":
    trainloader,testloader = cifar10_loader(batch_size=128,data_path="../data")
    CUDA =True

    model = AlexNet().cuda()
    layers = [ "conv1","conv2","conv3","conv4","conv5" ]


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

            loss= get_loss(labels,preds,model)
            loss.backward()
            optimizer.step()
            # model.apply(binarizer)
            writer.add_scalar('loss/train', loss , global_step)

            global_step+=1
            print(f"loss:{loss}")


        if epoch%1==0:
            # torch.save(model.state_dict(), f"./checkpoint/model_{epoch}")

            test_acc = evaluate(model,testloader,cuda = CUDA)
            train_acc = evaluate(model,trainloader,cuda = CUDA)
            print(f"Epoch {epoch} complete:")
            print(f"\ttest Accuracy : {test_acc}")
            print(f"\ttrain Accuracy : {train_acc}")

            model.eval()
            test_acc = evaluate(model,testloader,cuda = CUDA)
            train_acc = evaluate(model,trainloader,cuda = CUDA)
            print(f"\teval_test Accuracy : {test_acc}")
            print(f"\teval_train Accuracy : {train_acc}")
            model.train()

    print('Finished Training')