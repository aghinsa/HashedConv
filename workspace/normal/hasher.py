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

from scipy.cluster.vq import kmeans,vq,whiten
from numpy import random

class HashedConv(nn.Conv2d):
	def init_hasher(self,hasher):
		self.init_hasher = hasher
	def forward(self,x):
		if not self.training:
			return super().forward(x)

		new_weight,prob,bins = self.hasher(self.weight)
		self.weight = new_weight
		return self.conv2d_forward(x,new_weight)

def get_conv_weights(model,cuda=True):
	weights = []
	biases = []
	names = []
	ip_sizes = []
	for name,weight in model.named_parameters():
		if "conv" in name:
			if "weight" in name:
				names.append(name.split('.')[0])
				ip_sizes.append(weight.size())
				weights.append(weight.clone().detach().cuda())
			else:
				biases.append(weight.clone().detach().cuda())


	return weights,biases,ip_sizes,names

def set_conv_weights(model,weights,biases,names):
	for w,b,n in zip(weights,biases,names):
		layer = getattr(model,n)
		setattr(layer,"weight",nn.Parameter(w))
		# setattr(layer,"bias",b)
	return model


class Hasher(nn.Module):
	def __init__(self,n_bins):
		super().__init__()
		self.n_bins = n_bins
		self.pretrained = False

		self.model_data = None
		self.training_data =None

		self.encoder = nn.Sequential(
			nn.Conv1d(1,16,1),
			nn.BatchNorm1d(16),
			nn.ReLU(),
			nn.Conv1d(16,n_bins,1),
			nn.BatchNorm1d(n_bins),
			nn.ReLU(),
			nn.Conv1d(n_bins,2*n_bins,1),
			nn.BatchNorm1d(2*n_bins),
			nn.ReLU(),
			nn.Conv1d(2*n_bins,2*n_bins,1),
			nn.BatchNorm1d(2*n_bins),
			nn.ReLU(),
		)

		self.bins_coder = nn.Sequential(
			nn.Linear(2*n_bins,2*n_bins),
			nn.ReLU(),
			nn.Linear(2*n_bins,n_bins)
		)
		self.classifier_conv = nn.Sequential(
			nn.Conv1d(2*n_bins,4*n_bins,1),
			nn.ReLU(),
			nn.Conv1d(4*n_bins,2*n_bins,1),
			nn.ReLU(),
		)

		self.classifier = nn.Sequential(
			nn.Linear(2*n_bins,2*n_bins),
			nn.ReLU(),
			nn.Linear(2*n_bins,n_bins),
			# nn.Softmax(dim=1),

		)


	def forward(self,weights):
		w1 = weights.reshape(-1,1)
		x = torch.unsqueeze(w1,2)
		x = self.encoder(x)

		prob = self.classifier_conv(x)
		prob = prob.squeeze()
		prob = self.classifier(prob)

		bins = self.bins_coder(x.squeeze())
		bins = bins.squeeze().mean(dim=0) # 16

		P = 1 + torch.pow((w1-bins),2)
		P = 1/P
		P = P/P.sum(1).unsqueeze(1)

		# print(f"bins : {bins.shape}")
		# print(f"P : {P.shape}")
		x_pred = (bins * P).sum(dim=1)
		# print(f"x_pred : {x_pred.shape}")
		x_pred = x_pred.reshape(weights.size())

		return x_pred,prob,bins

	def set_extras(self,extras):
		self.extras = extras

def klloss(p,q):
	loss = p*( p.log() - q.log() )
	loss = loss.mean()
	return loss

def get_labels_and_bins(n_bins,weight):
	np.random.seed(1024)
	x = weight.cpu()
	bins,_= kmeans(x,n_bins)
	# bins = torch.rand(16).cpu().numpy()
	bins = bins.squeeze()
	labels = np.argmin(np.abs(x-bins),1)
	labels = labels.cuda()
	bins = bins.reshape(-1,1)
	bins = torch.from_numpy(bins).float().cuda()
	bins.requires_grad = False
	reconstructed = bins[labels]
	np.random.seed()
	return labels,bins,reconstructed

if __name__ ==  "__main__":


	model = load_alexnet()
	model.cuda()

	weights,biases,ip_sizes,names = get_conv_weights(model)
	labels = []
	bins_og = []
	rcs = []
	for w in weights:
		l,b,rc = get_labels_and_bins(16,w.reshape(-1,1))
		labels.append(l)
		bins_og.append(b)
		rcs.append(rc)

	training_data = {
		"w" : weights,
		"label":labels,
		"bin_og":bins_og,
		"ip_size":ip_sizes,
		"rc":rcs
	}

	hasher = Hasher(16)
	hasher.cuda()

	optimizer = optim.Adam(hasher.parameters(),lr=0.0001)
	global_step = 0

	kldiv = nn.KLDivLoss(reduction="batchmean")
	mse = nn.MSELoss(reduction = "mean")
	ce = nn.NLLLoss(reduction="mean")

	writer = SummaryWriter()

	load_from = "./checkpoint/pretrained_hasher_no"
	# load_from = None
	if load_from is not None:
		hasher.load_state_dict(torch.load(load_from))


	for epoch in range(1,1000):
		total_loss = 0
		total_up_loss = 0
		total_class_loss = 0
		total_bin_loss = 0
		for x,label,bin_og,ip_size,rc in zip(*training_data.values()):
			x_pred,prob,bins = hasher.forward(x)

			# P = 1 + torch.pow((x_pred.reshape(-1,1)-bins),2)
			# P = 1/P
			# P = P/P.sum(1).unsqueeze(1)

			rc = rc.reshape(x.size())
			up_loss = mse(x_pred,x) + 0.5 * mse(x_pred,rc)
			bin_loss = mse(bins.squeeze(),bin_og.squeeze())

			# class_loss = ce(prob.log(),label) + kldiv(prob.log(),P)

			total_up_loss +=up_loss
			# total_class_loss +=class_loss
			total_bin_loss +=bin_loss

			# loss = up_loss + class_loss + bin_loss
			loss = up_loss  + bin_loss
			# loss = bin_loss
			total_loss+=loss

			global_step+=1
			writer.add_scalar("loss/up_loss",up_loss)
			# writer.add_scalar("loss/class_loss",class_loss)
			writer.add_scalar("loss/bin_loss",bin_loss)



			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print(f"up_loss:{total_up_loss}")
		# print(f"class_loss:{total_class_loss}")
		print(f"bin_loss:{total_bin_loss}")

		print(f"{epoch}:loss:{total_loss}")
		if (epoch%50==0):
			if (total_loss != total_loss).any():
				raise(Exception(f"NaN encountered in epoch : {epoch}"))
			torch.save(hasher.state_dict(), f"./checkpoint/pretrained_hasher_no")

	print(f"pretrain  complete")


