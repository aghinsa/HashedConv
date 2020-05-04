import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
from  itertools import chain

from blocks import ProjectDown,ProjectUp
from hashnet import HashNet,CentroidNet
from utils import evaluate,cifar10_loader

from alexnet import load_alexnet
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision

def get_conv_weights(model,cuda=True):
	weights = []
	biases = []
	names = []
	ip_sizes = set()
	for name,weight in model.named_parameters():
		if "conv" in name:
			if "weight" in name:
				names.append(name.split('.')[0])
				ip_sizes.add(weight.size())
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
	def __init__(self,ip_sizes,n_bins):
		super().__init__()
		self.ups = { ip_size : ProjectUp(ip_size).cuda() for ip_size in ip_sizes}
		self.downs = { ip_size : ProjectDown(ip_size).cuda() for ip_size in ip_sizes}

		self.hashnet = HashNet(n_bins) # for bins
		self.centroid_net = CentroidNet(n_bins) # for probs
		self.n_bins = n_bins
		self.pretrained = False
	def init(self):
		load_from=f"./checkpoint/pretrained_hasher_no"
		self.load_state_dict(torch.load(load_from))
		self.pretrained = True

	def forward(self,x):
		# expects layer weifhts
		up = self.ups[x.size()]
		down = self.downs[x.size()]

		x_down = down(x)
		x_up = up(x_down)

		bins = self.hashnet(x_down)
		bins = bins.view(1,-1)

		x_flat = x.view(-1,1)
		bins_rect = bins.repeat(x_flat.size()[0],1)
		x_bin = torch.cat( [ bins_rect,x_flat],dim=1 )
		prob = self.centroid_net(x_bin)


		if  self.training:
			new_weight = torch.mean(bins_rect*prob,dim=1)
			Q = x_flat - bins
			Q = torch.pow(Q,1)+1
			Q = 1/Q
			Q = Q/torch.unsqueeze( torch.sum(Q,dim=1) ,1 )

		else:
			_,idx = torch.max(prob,dim=1)
			bins = torch.squeeze(bins)
			new_weight = bins[idx]
			Q = 0
		new_weight = new_weight.view(x.size())

		bins = bins.reshape(-1,1).double()
		return new_weight,x_up,Q,prob,bins




def hash_loss(original_weight,new_weight,x_up,P,Q,prob):
	kl = nn.KLDivLoss(reduction="batchmean")
	mse = nn.MSELoss(reduction = "mean")
    # kl expects target as prob and inputs as log prob

	kl1 = 0
	# kl1 += kl(torch.log(Q),P)
	# print(f"kl1:{kl1}")
	# kl1 += kl(torch.log(Q),prob)
	# print(f"kl2:{kl1}")


	rl=0
	rl+=mse(x_up,original_weight)
	# print(f"rl1 : {rl}")
	rl+=mse(new_weight,original_weight)
	# print(f"rl2 : {rl}")

	loss = kl1+rl

	return loss



if __name__ ==  "__main__":


	model = load_alexnet()
	model.cuda()

	weights,biases,ip_sizes,names = get_conv_weights(model)
	hasher = Hasher(ip_sizes,16)
	hasher.cuda()

	# hasher.pretrain(model,load_from=None)
	hasher.pretrain(model,load_from=f"./checkpoint/pretrained_hasher_no")
	# optimizer = optim.Adam(hasher.parameters(),lr=0.0001)
	# global_step = 0

	# for epoch in range(1,n_epochs+1):
	# 	for weight in chain(*[weights]):
	# 		new_weight,x_up,P,Q,prob = hasher(weight)
	# 		loss = hash_loss(weight,new_weight,x_up,P,Q,prob)
	# 		print(f"loss:{loss}")
	# 		optimizer.zero_grad()
	# 		loss.backward()
	# 		optimizer.step()
	# 	print(f"epoch : {epoch} complete")
	# 	if epoch%20 == 0:
	# 		hasher.eval()
	# 		new_weights = []
	# 		new_biases = []
	# 		for weight in weights:
	# 			new_weight,x_up,P,Q,prob = hasher(weight)
	# 			new_weights.append(new_weight)
	# 		# for bias in biases:
	# 		# 	new_bias,x_up,P,Q,prob = hasher(bias)
	# 		# 	new_biases.append(new_bias)

	# 		set_conv_weights(model,new_weights,new_biases,names)
	# 		test_acc = evaluate(model,testloader,cuda=True)
	# 		train_acc = evaluate(model,trainloader,cuda=True)

	# 		print(f"Epoch {epoch} complete:")
	# 		print(f"\ttest Accuracy : {test_acc}")
	# 		print(f"\ttrain Accuracy : {train_acc}")


	# 		torch.save(hasher.state_dict(), f"./checkpoint/hasher_{epoch}")
	# 		hasher.train()



