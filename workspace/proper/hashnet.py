#source https://github.com/hydxqing/ReNet-pytorch-keras-chapter3/

#coding:utf-8
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import sys
from torch.autograd import gradcheck
import time
import math
import argparse


from blocks import ProjectDown,ProjectUp

"""
weights -> projection -> hashnet
"""

class HashNet(nn.Module):
	def __init__(self,n_bins):
		# inputs are [-1,64,16]
		super().__init__()
		self.n_bins = n_bins
		# self.encoder = nn.TransformerEncoderLayer(d_model=16, nhead=4)
		self.encoder = nn.Sequential(
			nn.Conv2d(1,n_bins,3),
			nn.BatchNorm2d(n_bins),
			nn.ReLU(),
			nn.Conv2d(n_bins,2*n_bins,3),
			nn.BatchNorm2d(2*n_bins),
			nn.ReLU(),
			nn.Conv2d(2*n_bins,n_bins,1),
			nn.BatchNorm2d(n_bins),
			nn.ReLU(),
		)
		dense1 = nn.Linear(64*16,256)
		dense2 = nn.Linear(256,64)
		dense3 = nn.Linear(64,64)
		dense4 = nn.Linear(64,n_bins)
		self.op = nn.Sequential(
			dense1,
			dense2,
			nn.BatchNorm1d(64),
			nn.ReLU(),
			dense3,
			dense4,
			)
	def forward(self,x):
		x = torch.unsqueeze(x,1)
		# print(f"coder in : {x.size()}")
		x = self.encoder(x)
		# print(f"coder out : {x.size()}")
		x = x.reshape(-1,64*16)
		x = self.op(x)
		x = torch.mean(x,dim=0) # [n_bins]
		# print(f"bin : {x.size()}")
		return x

class CentroidNet(nn.Module):
	def __init__(self,n_bins):
		super().__init__()
		self.op = nn.Sequential(
			nn.Linear(n_bins+1,2*n_bins),
			nn.Linear(2*n_bins,n_bins),
			nn.Linear(n_bins,n_bins),
			nn.Softmax(dim=1)
		)
	def forward(self,x):
		x = self.op(x)
		return x

class HashedConv2d(nn.Module):
	def __init__(self,*args,**kwargs):
		super().__init__()
		self.hashnet = kwargs["hashnet"]
		self.n_bins = self.hashnet.n_bins
		self.centroid_net = kwargs["centroid_net"]
		kwargs.pop("hashnet")
		kwargs.pop("centroid_net")

		self.conv = nn.Conv2d(*args,**kwargs)

		self.project_down = ProjectDown(self.conv.weight.size())
		self.project_up = ProjectUp(self.conv.weight.size())

		self.conv_hashed = nn.Conv2d(*args,**kwargs)
		self.kl = nn.KLDivLoss(reduction="mean")

	def forward(self,X):
		weights = self.conv.weight

		weights_projection = self.project_down(weights)
		weights_up = self.project_up(weights_projection)

		bins = self.hashnet(weights_projection)
		bins = bins.view(1,-1)
		weights_re = weights.view(-1,1)

		weights_re = torch.cat( [ bins.repeat(weights_re.size()[0],1),weights_re],dim=1 )

		prob = self.centroid_net(weights_re)

		if self.training:
			new_weights = prob*bins
			new_weights = torch.mean(new_weights,dim=1)

			weights_re = weights.view(-1,1)

			Q = weights_re - bins
			Q = torch.pow(Q,1)+1
			Q = 1/Q
			Q = Q/torch.unsqueeze( torch.sum(Q,dim=1) ,1 )


			f = torch.sum(Q,dim=0)
			P = torch.pow(Q,2)/f
			P = P/torch.unsqueeze( torch.sum(P,dim=1) , 1)



		else :
			_,idx = torch.max(prob,dim=1)
			bins = torch.squeeze(bins)
			new_weights = bins[idx]
			P = 0
			Q = 0

		new_weights = new_weights.view(weights.size())
		# self.conv_hashed.weight = nn.Parameter(new_weights,requires_grad = False)
		self.conv_hashed.weight = nn.Parameter(new_weights)


		x = self.conv(X)
		x_hash = self.conv_hashed(X)
		# Q = torch.log(Q)
		# kl = self.kl(Q,prob)
		# print(f"\n\n {prob.size()}")
		# print(f"\n\n {Q.size()}")

		prob = 0
		rl = nn.MSELoss(reduction="mean")(weights,weights_up)
		return (x,x_hash,P,Q,prob,rl)


if __name__ == "__main__":
	x = torch.randn(2,16,32,32)
	n_bins = 16
	hashnet = HashNet(n_bins)
	centroid_net = CentroidNet(n_bins)
	model = HashedConv2d(16,54,3,hashnet = hashnet,centroid_net = centroid_net  )

	# model.eval()
	y,yhash,P,Q,prob,rl=model(x)
	print(f"y : {y.size()}")
	print(f"loss : {prob.size()}")


