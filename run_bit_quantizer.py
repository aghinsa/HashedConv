import sys

sys.path.insert(0, ".")

import torch

from alexnet import AlexNet
from pytorch_resnet.resnet import resnet32, resnet56
from quantizer import BitQuantizer
from utils import cifar10_loader, evaluate

# model = AlexNet()
# model.load_state_dict(torch.load("./alexnet_pretrained"))

model = resnet32()
checkpoint = torch.load("./pytorch_resnet/pretrained_models/resnet32-d509ac18.th")
state = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
model.load_state_dict(state)

trainloader, testloader = cifar10_loader(256, "../data")

model.cuda()


n_fs = 32
n_bits = 6
n_iter = 50
avoid = []
bit_quantizer = BitQuantizer(model, n_fs, n_bits, avoid=avoid)
bit_quantizer.train_hash_functions(n_iter=n_iter)
hashed_model = bit_quantizer.get_hashed_model()


hashed_model.cuda()


# Evaluating model before hashing
train_accracy = evaluate(model, trainloader, cuda=True)
test_accracy = evaluate(model, testloader, cuda=True)

print(f"Train accuracy before hashing: {train_accracy} ")
print(f"Test accuracy before hashing: {test_accracy} ")

# Evaluating hashed model
train_accracy = evaluate(hashed_model, trainloader, cuda=True)
test_accracy = evaluate(hashed_model, testloader, cuda=True)

print(f"Train accuracy after hashing: {train_accracy} ")
print(f"Test accuracy after hashing: {test_accracy} ")
