# imports
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class HashedConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.buckets = kwargs["buckets"]
        kwargs.pop("buckets")
        self.hashed_weights = torch.randn(self.buckets)
        self.conv = nn.Conv2d(*args, **kwargs)
        self.hashed_conv = nn.Conv2d(*args, **kwargs)

    def forward(self, x):
        out = self.conv(x)
        max_weight = torch.max(self.conv.weight)
        min_weight = torch.min(self.conv.weight)
        size = self.hashed_conv.weight.size()
        weight = self.conv.weight.clone()
        for i in range(self.buckets):
            threshold = (i+1)*((max_weight - min_weight)/self.buckets) + min_weight
            for m in range(size[0]):
                for c in range(size[1]):
                    for r in range(size[2]):
                        for s in range(size[3]):
                            if (weight[m][c][r][s] > threshold):
                                weight[m][c][r][s] = self.hashed_weights[i]
        self.hashed_conv.weight = nn.Parameter(weight)
        hashed_out = self.hashed_conv(x)
        return out, hashed_out

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        #self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1 = HashedConv2d(3, 6, 5, buckets = 16)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2 = HashedConv2d(6, 16, 5, buckets = 16)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #out = F.relu(self.conv1(x))
        #out = F.max_pool2d(out, 2)
        #out = F.relu(self.conv2(out))
        #out = F.max_pool2d(out, 2)
        out, hc1 = self.conv1(x)
        c1 = out.clone()
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out, hc2 = self.conv2(out)
        c2 = out.clone()
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out, c1, hc1, c2, hc2
    
if __name__ == "__main__":
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=0)
    
    # constant for classes
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    net = LeNet()
    
    criterion = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    for epoch in range(2):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs, c1, hc1, c2, hc2 = net(inputs)
            loss = criterion(outputs, labels) + mse(c1, hc1) + mse(c2, hc2)
            print(f"loss:{loss}")
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    
    print('Finished Training')
                
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs, _ = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))