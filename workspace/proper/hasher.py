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
			nn.Conv2d(3,16,5),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.Conv2d(16,n_bins,5),
			nn.BatchNorm2d(n_bins),
			nn.ReLU(),
			nn.Conv2d(n_bins,2*n_bins,5),
			nn.BatchNorm2d(2*n_bins),
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
			nn.Softmax(dim=1),

		)


	def forward(self,x):
		x = self.encoder(x)
		return x


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

	hasher = Hasher(16)
	hasher.cuda()
	x = torch.rand(2,3,32,32).cuda()
	print(hasher(x).size())

