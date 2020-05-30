# imports
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Hashed import HashedConv2d

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = HashedConv2d(3, 6, 5, buckets = 16)
        self.conv2 = HashedConv2d(6, 16, 5, buckets = 16)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out, hc1, out_hash = self.conv1(x, x)
        c1 = out.clone()
        out = F.relu(out)
        out_hash = F.relu(out_hash)
        out = F.max_pool2d(out, 2)
        out_hash = F.max_pool2d(out_hash, 2)
        
        out, hc2, out_hash = self.conv2(out, out_hash)
        c2 = out.clone()
        out = F.relu(out)
        out_hash = F.relu(out_hash)
        out = F.max_pool2d(out, 2)
        out_hash = F.max_pool2d(out_hash, 2)
        
        out = out.view(out.size(0), -1)
        out_hash = out_hash.view(out_hash.size(0), -1)
        
        out = F.relu(self.fc1(out))
        out_hash = F.relu(self.fc1(out_hash))
        out = F.relu(self.fc2(out))
        out_hash = F.relu(self.fc2(out_hash))
        out = self.fc3(out)
        out_hash = self.fc3(out_hash)
        return out, out_hash, c1, hc1, c2, hc2
    
if __name__ == "__main__":
    # global variables
    BATCH_SIZE = 100
    EPOCH = 50
    LAMBDA = 0.01
    GAMMA = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    net = LeNet()
    
    criterion = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    optimizer = optim.Adam(net.parameters())
    
    net.to(DEVICE)
    print(net.conv1.hash_value)
    
    for epoch in range(EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        running_criterion = 0.0
        running_criterion_hash = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
    
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            out, out_hash, c1, hc1, c2, hc2 = net(inputs)
            loss = criterion(out, labels) + GAMMA*(criterion(out_hash, labels)) + LAMBDA*(mse(c1, hc1) + mse(c2, hc2))
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            running_criterion += criterion(out, labels).item()
            running_criterion_hash += criterion(out_hash, labels).item()
            if i % 10 == 9:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f criterion: %.3f hash_criterion: %.3f' %
                    (epoch+1, i+1, running_loss/1, running_criterion/1, running_criterion_hash/1))
            running_loss = 0.0
            running_criterion = 0.0
            running_criterion_hash = 0.0
    
    print('Finished Training')
    print(net.conv1.hash_value)
                
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            out, out_hash, c1, hc1, c2, hc2 = net(images)
            _, predicted = torch.max(out_hash.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))