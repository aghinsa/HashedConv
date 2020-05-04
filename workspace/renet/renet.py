#source https://github.com/hydxqing/ReNet-pytorch-keras-chapter3/

#coding:utf-8
import torch
import torchvision
import torchvision.transforms as transforms
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

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage

class ReNet(nn.Module):
	def __init__(self, receptive_filter_size, hidden_size,num_lstm_layers=1,n_bins=16,device=torch.device("cpu")):

		super().__init__()

		self.receptive_filter_size = receptive_filter_size
		# Todo change 1 to arbitary
		# possible another rnn in depth
		self.input_size1 = receptive_filter_size * receptive_filter_size * 1
		self.input_size2 = hidden_size * 2
		self.hidden_size = hidden_size
		self.num_lstm_layers = num_lstm_layers
		self.device = device

		# vertical rnns
		self.rnn1 = nn.LSTM(self.input_size1, self.hidden_size,num_lstm_layers, dropout = 0.2)
		self.rnn2 = nn.LSTM(self.input_size1, self.hidden_size,num_lstm_layers, dropout = 0.2)

		# horizontal rnns
		self.rnn3 = nn.LSTM(self.input_size2, self.hidden_size,num_lstm_layers, dropout = 0.2)
		self.rnn4 = nn.LSTM(self.input_size2, self.hidden_size,num_lstm_layers, dropout = 0.2)


		self.n_bins=n_bins

		self.batch_norm1 = torch.nn.BatchNorm2d(self.hidden_size*2)
		self.batch_norm2 = torch.nn.BatchNorm2d(64) #check if this is correct

		#TODO confirm usage of 8
		self.adpt_avg = nn.AdaptiveAvgPool3d((self.hidden_size*2,8,8))
		self.adpt_avg2d = nn.AdaptiveAvgPool2d((1,3*n_bins))
		self.conv1 = nn.Conv2d(self.hidden_size*2,64,3,3 )
		self.fc1 = nn.Linear(256,64)
		self.fc2 = nn.Linear(64,n_bins)
		self.fc3 = nn.Linear(3*n_bins,n_bins)

		self.cls_fc1 = nn.Linear(n_bins+1,n_bins)
		self.cls_fc2 = nn.Linear(n_bins,n_bins)
		self.cls_fc3 = nn.Linear(n_bins,n_bins)


		self.softmax = nn.Softmax(dim=1)



	def get_image_patches(self, X, receptive_filter_size):
		"""
		creates image patches based on the dimension of a receptive filter
		"""
		image_patches = []
		_, X_channel, X_height, X_width= X.size()


		for i in range(0, X_height, receptive_filter_size):
			for j in range(0, X_width, receptive_filter_size):
				X_patch = X[:, :, i: i + receptive_filter_size, j : j + receptive_filter_size]
				image_patches.append(X_patch)

		image_patches_height = (X_height // receptive_filter_size)
		image_patches_width = (X_width // receptive_filter_size)


		image_patches = torch.stack(image_patches)
		image_patches = image_patches.permute(1, 0, 2, 3, 4)

		image_patches = image_patches.contiguous().view(-1, image_patches_height, image_patches_height, receptive_filter_size * receptive_filter_size * X_channel)

		return image_patches



	def get_vertical_rnn_inputs(self, image_patches, forward):
		"""
		creates vertical rnn inputs in dimensions 
		(num_patches, batch_size, rnn_input_feature_dim)
		num_patches: image_patches_height * image_patches_width
		"""
		vertical_rnn_inputs = []
		_, image_patches_height, image_patches_width, feature_dim = image_patches.size()

		if forward:
			for i in range(image_patches_height):
				for j in range(image_patches_width):
					vertical_rnn_inputs.append(image_patches[:, j, i, :])

		else:
			for i in range(image_patches_height-1, -1, -1):
				for j in range(image_patches_width-1, -1, -1):
					vertical_rnn_inputs.append(image_patches[:, j, i, :])

		vertical_rnn_inputs = torch.stack(vertical_rnn_inputs).cuda()


		return vertical_rnn_inputs



	def get_horizontal_rnn_inputs(self, vertical_feature_map, image_patches_height, image_patches_width, forward):
		"""
		creates vertical rnn inputs in dimensions 
		(num_patches, batch_size, rnn_input_feature_dim)
		num_patches: image_patches_height * image_patches_width
		"""
		horizontal_rnn_inputs = []

		if forward:
			for i in range(image_patches_height):
				for j in range(image_patches_width):
					horizontal_rnn_inputs.append(vertical_feature_map[:, i, j, :])
		else:
			for i in range(image_patches_height-1, -1, -1):
				for j in range(image_patches_width -1, -1, -1):
					horizontal_rnn_inputs.append(vertical_feature_map[:, i, j, :])

		horizontal_rnn_inputs = torch.stack(horizontal_rnn_inputs).cuda()

		return horizontal_rnn_inputs


	def renet_layer(self, X,hidden):

		"""ReNet """
		# divide input input image to image patches
		image_patches = self.get_image_patches(X, self.receptive_filter_size)
		_, image_patches_height, image_patches_width, feature_dim = image_patches.size()
		# process vertical rnn inputs
		vertical_rnn_inputs_fw = self.get_vertical_rnn_inputs(image_patches, forward=True)
		vertical_rnn_inputs_rev = self.get_vertical_rnn_inputs(image_patches, forward=False)

		# extract vertical hidden states
		vertical_forward_hidden, vertical_forward_cell = self.rnn1(vertical_rnn_inputs_fw, hidden)
		vertical_reverse_hidden, vertical_reverse_cell = self.rnn2(vertical_rnn_inputs_rev, hidden)

		# create vertical feature map
		vertical_feature_map = torch.cat((vertical_forward_hidden, vertical_reverse_hidden), 2)
		vertical_feature_map =  vertical_feature_map.permute(1, 0, 2)

		# reshape vertical feature map to (batch size, image_patches_height, image_patches_width, hidden_size * 2)
		vertical_feature_map = vertical_feature_map.contiguous().view(-1, image_patches_width, image_patches_height, self.hidden_size * 2)
		vertical_feature_map.permute(0, 2, 1, 3)

		# process horizontal rnn inputs
		horizontal_rnn_inputs_fw = self.get_horizontal_rnn_inputs(vertical_feature_map, image_patches_height, image_patches_width, forward=True)
		horizontal_rnn_inputs_rev = self.get_horizontal_rnn_inputs(vertical_feature_map, image_patches_height, image_patches_width, forward=False)

		# extract horizontal hidden states
		horizontal_forward_hidden, horizontal_forward_cell = self.rnn3(horizontal_rnn_inputs_fw, hidden)
		horizontal_reverse_hidden, horizontal_reverse_cell = self.rnn4(horizontal_rnn_inputs_rev, hidden)

		# create horiztonal feature map[64,1,320]
		horizontal_feature_map = torch.cat((horizontal_forward_hidden, horizontal_reverse_hidden), 2)
		horizontal_feature_map =  horizontal_feature_map.permute(1, 0, 2)

		# flatten[1,64,640]
		output = horizontal_feature_map.contiguous().view(-1, image_patches_height , image_patches_width , self.hidden_size * 2)
		output=output.permute(0,3,1,2)#[1,640,8,8]

		return output

	def renet_forward(self,X):
		X_batch_size, X_channel, X_height, X_width= X.size()

		# Padding to nearest multiple of receptive field size
		if (not (X_height%self.receptive_filter_size==0)):
			height_pad = (self.receptive_filter_size - X_height%self.receptive_filter_size)
			width_pad = (self.receptive_filter_size - X_width%self.receptive_filter_size)

			X = F.pad(X,(0,height_pad,0,width_pad),mode = "replicate")

		xs=torch.unbind(X,dim=1)
		new_xs = []
		#this has to be explicitly moved to device
		#see https://github.com/pytorch/pytorch/issues/15272
		hidden = (
				torch.zeros(self.num_lstm_layers , X_batch_size, self.hidden_size,device=self.device),
				torch.zeros(self.num_lstm_layers , X_batch_size, self.hidden_size,device=self.device)
				)
		for x in xs:
			x = torch.unsqueeze(x,1)
			x = self.renet_layer(x,hidden)
			x = self.adpt_avg(x) # [batch_size,hidden*2,8,8]
			# x = self.batch_norm1(x)
			new_xs.append(x)

		return new_xs

	def forward(self,X):
		#list of length n_channels
		xs = self.renet_forward(X) # [batch_size,hidden*2,8,8]
		#hashing
		bins = []
		for x in xs:
			x = self.conv1(x) #[batch_size,64,2,2]
			x = self.batch_norm2(x)
			x = x.view(-1,64*4)
			bins.append(x)

		# need to have one layer from here in classifier
		bins = torch.stack(bins,1) # [n,c,64*4]
		bins = bins.view(-1,64*4) # [n*c,256]
		bins = self.fc1(bins) # [n*c,64]
		bins = self.fc2(bins) # [n*c,bins]
		bins=torch.unsqueeze(bins,0)
		bins =  self.adpt_avg2d(bins)

		bins = self.fc3(bins) #[1,1,bins]
		bins = torch.squeeze(bins) #[bins]


		X_col = X.reshape(-1,1)
		bins_repeated = bins.view(1,-1)
		bins_repeated = bins_repeated.repeat(X_col.size()[0],1)
		X_col = torch.cat( [X_col,bins_repeated],dim=1 )


		prob_mat = self.classifier(X_col)


		_,c_idx = torch.max(prob_mat,dim=1)

		centroids = bins[c_idx]


		Q = torch.unsqueeze(bins,0) - torch.unsqueeze(centroids,1)
		Q = 1+torch.pow(Q,2)
		Q = torch.pow(Q,-0.5)
		Q_denom = torch.sum(Q,dim=1)
		Q = Q/torch.unsqueeze(Q_denom,1)

		f=torch.unsqueeze(torch.sum(prob_mat,axis=0),0) #[1,n_bins]
		P = torch.pow(Q,2)/f
		P = P/torch.unsqueeze(torch.sum(P,1),1)

		if not self.training:
			hashed = centroids.view(X.size())
		else:
			hashed = torch.sum(prob_mat*bins,dim=1).view(X.size())
		return hashed,P,Q

	def classifier(self,x):
		x = self.cls_fc1(x)
		x = self.cls_fc2(x)
		x = self.cls_fc3(x)
		x = self.softmax(x)
		return x

if __name__ == "__main__":
	renet = ReNet(
			receptive_filter_size=4,
			hidden_size=16,
			num_lstm_layers=1,
			n_bins=8,
			device=torch.device("cuda"))
	renet.cuda()
	x = torch.rand(12,32,3,3).cuda()
	xer,p,q = renet.forward(x)
	print(f"X : {p.size()}")
