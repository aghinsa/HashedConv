import torch

from alexnet import AlexNet
from quantizer import BitQuantizer
from utils import cifar10_loader,evaluate

model = AlexNet()
model.load_state_dict(torch.load("./alexnet_pretrained"))
trainloader,testloader = cifar10_loader(256,"../data")

model.cuda()
train_accracy = evaluate(model,trainloader,cuda = True)
test_accracy = evaluate(model,testloader,cuda = True)

print(f"Train accuracy before hashing: {train_accracy} ")
print(f"Test accuracy before hashing: {test_accracy} ")

n_fs = 16
n_bits = 6
n_iter = 2
avoid =["conv1","conv2","conv3","conv4","classifier.4","classsifier.6"]
bit_quantizer = BitQuantizer(
                    model,
                    n_fs,
                    n_bits,
                    avoid=avoid
             )
bit_quantizer.train_hash_functions(n_iter = n_iter)
hashed_model = bit_quantizer.get_hashed_model()

hashed_model.cuda()
train_accracy = evaluate(hashed_model,trainloader,cuda = True)
test_accracy = evaluate(hashed_model,testloader,cuda = True)

print(f"Train accuracy after hashing: {train_accracy} ")
print(f"Test accuracy after hashing: {test_accracy} ")
