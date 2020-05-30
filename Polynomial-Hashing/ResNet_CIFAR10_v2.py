# imports
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from resnet import ResNet34
from Hashed import HashedConv2d

from collections import OrderedDict

def key_transform(key):
    key = key.replace('.downsample.0.', '.shortcut.conv.')
    key = key.replace('.downsample.1.', '.shortcut.bn.')
    return key

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# load model
res34 = ResNet34()

# pretrained model taken from - https://github.com/huyvnphan/PyTorch_CIFAR10
pretrained_dict = torch.load("./state_dicts/resnet34.pt")
new_dict = OrderedDict()
for key, value in pretrained_dict.items():
    new_key = key_transform(key)
    new_dict[new_key] = value

model_dict = res34.state_dict()
model_dict.update(new_dict)
res34.load_state_dict(model_dict)

# only update the parameters corresponding to hashing
update_params = []
for module in res34.modules():
    if(isinstance(module, HashedConv2d)):
        update_params.append({'params': module.a0})
        update_params.append({'params': module.a1})
        update_params.append({'params': module.a2})

# global variables
BATCH_SIZE = 100
EPOCH = 50
GAMMA = 100
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VERSION = 2

# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# datasets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=4)

# constant for classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = res34

criterion = nn.CrossEntropyLoss()
mse = nn.MSELoss()
optimizer = optim.Adam(update_params, lr=LR)

net.to(DEVICE)
loss_epoch = []
loss_epoch_hash = []


for epoch in range(EPOCH):  # loop over the dataset multiple times

    # running_loss = 0.0
    running_criterion = 0.0
    running_criterion_hash = 0.0
    pl_loss = 0.0
    pl_loss_hash = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        out, out_hash = net(inputs)
        loss = criterion(out_hash, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_criterion += criterion(out, labels).item()
        running_criterion_hash += criterion(out_hash, labels).item()
        pr = 100
        if i % pr == pr-1:    # print every 10 mini-batches
            print('[%d, %5d] criterion: %.5f hash_criterion: %.5f' %
                  (epoch+1, i+1, running_criterion, running_criterion_hash))
        
        pl_loss += running_criterion
        pl_loss_hash += running_criterion_hash
        running_criterion = 0.0
        running_criterion_hash = 0.0
    loss_epoch.append(pl_loss/(i+1))
    loss_epoch_hash.append(pl_loss_hash/(i+1))

print('Finished Training')

correct = 0
total = 0
correct_hash = 0
with torch.no_grad():
    for data in trainloader:
        images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        out, out_hash = net(images)
        _, predicted = torch.max(out.data, 1)
        _, predicted_hash = torch.max(out_hash.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        correct_hash += (predicted_hash == labels).sum().item()
train_acc = 100 * correct / total
train_acc_hash = 100 * correct_hash / total
print('Accuracy of the base network on train images: %d %%' % (
    train_acc))
print('Accuracy of the hashed network on train images: %d %%' % (
    train_acc_hash))
            
correct = 0
total = 0
correct_hash = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        out, out_hash = net(images)
        _, predicted = torch.max(out.data, 1)
        _, predicted_hash = torch.max(out_hash.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        correct_hash += (predicted_hash == labels).sum().item()

test_acc = 100 * correct / total
test_acc_hash = 100 * correct_hash / total
print('Accuracy of the base network on test images: %d %%' % (
    test_acc))
print('Accuracy of the hashed network on test images: %d %%' % (
    test_acc_hash))

# Saving and Plotting - create appropriate paths

PATH = './ResNet_v%d/ResNet_CIFAR10_v%d.pth' % (VERSION, VERSION) 
torch.save(net.state_dict(), PATH)

ep = np.arange(1,EPOCH+1)
plt.plot(ep, loss_epoch, label='Base Net')
plt.plot(ep, loss_epoch_hash, label='Hashed Net')
plt.axes()
plt.legend()
plt.xlabel('Number of Epochs')
plt.ylabel('Cross Entropy Loss')
plt.title('Loss on Training Data vs No. of Epochs')
plt.tight_layout()
# plt.xticks(np.arange(1, EPOCH+1))
plt.savefig('./ResNet_v%d/LC_loss_epoch_plot_%d.pdf' % (VERSION, VERSION), dpi=1200)
plt.close()