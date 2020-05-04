#ALEXNET
#source : https://github.com/icpm/pytorch-cifar10/blob/master/models/AlexNet.py

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

from renet import ReNet

NUM_CLASSES = 10

CUDA = True

if CUDA :
	DEVICE = torch.device("cuda")
else:
	DEVICE = torch.device("cpu")





class HashedConv2d(nn.Module):
	def __init__(self,*args,**kwargs):
		super().__init__()
		self.hashnet = kwargs["hashnet"]
		kwargs.pop("hashnet")
		self.conv = nn.Conv2d(*args,**kwargs)
		self.conv_hashed = nn.Conv2d(*args,**kwargs)

	def forward(self,X):
		weights = self.conv.weight
		if self.training:
			hashed_weights,P,Q = self.hashnet(weights)
			self.conv_hashed.weight = nn.Parameter(hashed_weights)

		x = self.conv(X)
		x_hash = self.conv_hashed(X)
		if self.training:
			return (x,x_hash,P,Q)
		else:
			return(x,x_hash,0,0)

class AlexNet(nn.Module):

	def __init__(self, num_classes=NUM_CLASSES):
		super(AlexNet, self).__init__()

		self.hashnet = ReNet(
			receptive_filter_size=4,
			hidden_size=16,
			num_lstm_layers=1,
			n_bins=8,
			device=DEVICE)


		self.conv1 = HashedConv2d(3, 64, kernel_size=3, stride=2, padding=1,hashnet=self.hashnet)
		self.ops1 = nn.Sequential(
						nn.ReLU(inplace=True),
				nn.MaxPool2d(kernel_size=2),
			)

		self.conv2 = HashedConv2d(64, 192, kernel_size=3, padding=1,hashnet=self.hashnet)
		self.ops2 = nn.Sequential(
						nn.ReLU(inplace=True),
				nn.MaxPool2d(kernel_size=2),
			)

		self.conv3 = HashedConv2d(192, 384, kernel_size=3, padding=1,hashnet=self.hashnet)
		self.ops3 = nn.ReLU(inplace=True)

		self.conv4 = HashedConv2d(384, 256, kernel_size=3, padding=1,hashnet=self.hashnet)
		self.ops4 = nn.ReLU(inplace=True)

		self.conv5 = HashedConv2d(256, 256, kernel_size=3, padding=1,hashnet=self.hashnet)
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
		ps=[]
		qs=[]

		x,xh,p,q = self.conv1(x)
		x = self.ops1(x)
		xh = self.ops1(xh)
		ps.append(p)
		qs.append(q)

		x,_,p,q = self.conv2(x)
		_,xh,_,_ = self.conv2(xh)
		x = self.ops2(x)
		xh = self.ops2(xh)
		ps.append(p)
		qs.append(q)

		x,_,p,q =  self.conv3(x)
		_,xh,_,_ = self.conv3(xh)
		x =  self.ops3(x)
		xh = self.ops3(xh)
		ps.append(p)
		qs.append(q)

		x,_,p,q =  self.conv4(x)
		_,xh,_,_ = self.conv4(xh)
		x =  self.ops4(x)
		xh = self.ops4(xh)
		ps.append(p)
		qs.append(q)

		x,_,p,q =  self.conv5(x)
		_,xh,_,_ = self.conv5(xh)
		x =  self.ops5(x)
		xh = self.ops5(xh)
		ps.append(p)
		qs.append(q)

		x = x.view(x.size(0), 256 * 2 * 2)
		x = self.classifier(x)
		x = self.softmax(x)

		xh = xh.view(xh.size(0), 256 * 2 * 2)
		xh = self.classifier(xh)
		xh = self.softmax(xh)


		return x,xh,ps,qs

if __name__ == "__main__":
	model = AlexNet()
	x= torch.rand(1,3,32,32)
	if CUDA:
		model.cuda()
		x=x.cuda()
	pred,pred_hashed,ps,qs = model(x)
	pass
