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
    def __init__(self, n_channels = 3, n_outputs = 10):
        super(LeNet, self).__init__()
        self.n_channels = n_channels
        self.n_outputs = n_outputs
        self.conv1 = HashedConv2d(n_channels, 6, 5, buckets = 16)
        self.conv2 = HashedConv2d(6, 16, 5, buckets = 16)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_outputs)
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc3.weight, gain=1.414)

    def forward(self, x):
        out, out_hash = self.conv1(x, x)
        out = F.relu(out)
        out_hash = F.relu(out_hash)
        out = F.max_pool2d(out, 2)
        out_hash = F.max_pool2d(out_hash, 2)
        
        out, out_hash = self.conv2(out, out_hash)
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
        return out, out_hash

if __name__ == "__main__":
    # global variables
    BATCH_SIZE = 50
    EPOCH = 1
    LAMBDA = 1
    GAMMA = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transforms
    transform = transforms.Compose([transforms.ToTensor()])

    # datasets
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                            download=True, transform=transform)

    # dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=4)

    # constant for classes
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet(n_channels=1)

    criterion = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    net.to(DEVICE)

    for epoch in range(EPOCH):  # loop over the dataset multiple times
        running_criterion = 0.0
        running_criterion_hash = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            out, out_hash = net(inputs)
            loss = criterion(out, labels) + GAMMA*(criterion(out_hash, labels))
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_criterion += criterion(out, labels).item()
            running_criterion_hash += criterion(out_hash, labels).item()
            pr = 100
            if i % pr == pr-1:    # print every 10 mini-batches
                print('[%d, %5d] criterion: %.5f hash_criterion: %.5f' %
                    (epoch+1, i+1, running_criterion, running_criterion_hash))
            running_criterion = 0.0
            running_criterion_hash = 0.0

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

    print('Accuracy of the base network on train images: %d %%' % (
        100 * correct / total))
    print('Accuracy of the hashed network on train images: %d %%' % (
        100 * correct_hash / total))
                
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

    print('Accuracy of the base network on test images: %d %%' % (
        100 * correct / total))
    print('Accuracy of the hashed network on test images: %d %%' % (
        100 * correct_hash / total))