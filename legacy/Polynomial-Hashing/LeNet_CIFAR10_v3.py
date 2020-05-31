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
from Hashed import HashedLinear

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class LeNet(nn.Module):
    def __init__(self, n_channels=3, n_outputs=10):
        super(LeNet, self).__init__()
        self.n_channels = n_channels
        self.n_outputs = n_outputs
        self.conv1 = HashedConv2d(n_channels, 6, 5, buckets=16)
        self.conv2 = HashedConv2d(6, 16, 5, buckets=16)
        self.fc1 = HashedLinear(16 * 5 * 5, 120, buckets=16)
        self.fc2 = HashedLinear(120, 84, buckets=16)
        self.fc3 = nn.Linear(84, n_outputs)
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

        out, out_hash = self.fc1(out, out_hash)
        out = F.relu(out)
        out_hash = F.relu(out_hash)

        out, out_hash = self.fc2(out, out_hash)
        out = F.relu(out)
        out_hash = F.relu(out_hash)

        out = self.fc3(out)
        out_hash = self.fc3(out_hash)
        return out, out_hash


if __name__ == "__main__":
    # global variables
    BATCH_SIZE = 100
    EPOCH = 100
    GAMMA = 10
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VERSION = 2

    # transforms
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    # transform = transforms.Compose([transforms.ToTensor()])

    # datasets
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # dataloaders
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # constant for classes
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    net = LeNet()
    # net = ResNet34()

    criterion = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)

    net.to(DEVICE)

    loss_epoch = []
    loss_batch = []
    loss_epoch_hash = []
    loss_batch_hash = []
    for epoch in range(EPOCH):  # loop over the dataset multiple times

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
            loss = criterion(out, labels) + GAMMA * (criterion(out_hash, labels))
            loss.backward()
            optimizer.step()

            # print statistics
            running_criterion += criterion(out, labels).item()
            running_criterion_hash += criterion(out_hash, labels).item()
            pr = 100
            if i % pr == pr - 1:  # print every 10 mini-batches
                print(
                    "[%d, %5d] criterion: %.5f hash_criterion: %.5f"
                    % (epoch + 1, i + 1, running_criterion, running_criterion_hash)
                )
                loss_batch.append(running_criterion)
                loss_batch_hash.append(running_criterion_hash)

            pl_loss += running_criterion
            pl_loss_hash += running_criterion_hash

            running_criterion = 0.0
            running_criterion_hash = 0.0
        loss_epoch.append(pl_loss / (i + 1))
        loss_epoch_hash.append(pl_loss_hash / (i + 1))

    print("Finished Training")

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

    print(
        "Accuracy of the base network on train images: %d %%" % (100 * correct / total)
    )
    print(
        "Accuracy of the hashed network on train images: %d %%"
        % (100 * correct_hash / total)
    )

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

    print(
        "Accuracy of the base network on test images: %d %%" % (100 * correct / total)
    )
    print(
        "Accuracy of the hashed network on test images: %d %%"
        % (100 * correct_hash / total)
    )

    # Saving and Plotting - create appropriate paths

    PATH = "./LeNet_v2/LeNet_CIFAR10_v%d.pth" % (VERSION)
    torch.save(net.state_dict(), PATH)

    ep = np.arange(1, EPOCH + 1)
    plt.plot(ep, loss_epoch, label="Base Net")
    plt.plot(ep, loss_epoch_hash, label="Hashed Net")
    plt.axes()
    plt.legend()
    plt.xlabel("Number of Epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Loss on Training Data vs No. of Epochs")
    plt.tight_layout()
    # plt.xticks(np.arange(1, EPOCH+1))
    plt.savefig("./LeNet_v2/LC_loss_epoch_plot_%d.pdf" % (VERSION), dpi=1200)
    plt.close()

    mb = np.arange(1, int(EPOCH * 50000 / (BATCH_SIZE * pr)) + 1)
    plt.plot(mb, loss_batch, label="Base Net")
    plt.plot(mb, loss_batch_hash, label="Hashed Net")
    plt.axes()
    plt.legend()
    plt.xlabel("Minibatches")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Loss on Training Data vs Minibatches")
    plt.tight_layout()
    # plt.xticks(np.arange(1, EPOCH+1))
    plt.savefig("./LeNet_v2/LC_loss_batch_plot_%d.pdf" % (VERSION), dpi=1200)
    plt.close()

    cw1 = torch.flatten(net.conv1.weight).cpu().detach().numpy()
    hw1 = torch.flatten(net.conv1.hash_weight).cpu().detach().numpy()
    num_bins = 16
    n, bins, patches = plt.hist(cw1, num_bins, facecolor="blue", alpha=0.5)
    plt.title("Histogram of conv1 weights")
    plt.savefig("./LeNet_v2/conv1_weight_histogram_%d.pdf" % (VERSION), dpi=1200)
    plt.close()
    n, bins, patches = plt.hist(hw1, num_bins, facecolor="blue", alpha=0.5)
    plt.title("Histogram of conv1 hashed weights")
    plt.savefig("./LeNet_v2/conv1_hash_weight_histogram_%d.pdf" % (VERSION), dpi=1200)
    plt.close()

    cw2 = torch.flatten(net.conv2.weight).cpu().detach().numpy()
    hw2 = torch.flatten(net.conv2.hash_weight).cpu().detach().numpy()
    num_bins = 16
    n, bins, patches = plt.hist(cw2, num_bins, facecolor="blue", alpha=0.5)
    plt.title("Histogram of conv2 weights")
    plt.savefig("./LeNet_v2/conv2_weight_histogram_%d.pdf" % (VERSION), dpi=1200)
    plt.close()
    n, bins, patches = plt.hist(hw2, num_bins, facecolor="blue", alpha=0.5)
    plt.title("Histogram of conv2 hashed weights")
    plt.savefig("./LeNet_v2/conv2_hash_weight_histogram_%d.pdf" % (VERSION), dpi=1200)
    plt.close()

    fcw1 = torch.flatten(net.fc1.weight).cpu().detach().numpy()
    fhw1 = torch.flatten(net.fc1.hash_weight).cpu().detach().numpy()
    num_bins = 16
    n, bins, patches = plt.hist(fcw1, num_bins, facecolor="blue", alpha=0.5)
    plt.title("Histogram of fc1 weights")
    plt.savefig("./LeNet_v2/fc1_weight_histogram_%d.pdf" % (VERSION), dpi=1200)
    plt.close()
    n, bins, patches = plt.hist(fhw1, num_bins, facecolor="blue", alpha=0.5)
    plt.title("Histogram of fc1 hashed weights")
    plt.savefig("./LeNet_v2/fc1_hash_weight_histogram_%d.pdf" % (VERSION), dpi=1200)
    plt.close()

    fcw2 = torch.flatten(net.fc2.weight).cpu().detach().numpy()
    fhw2 = torch.flatten(net.fc2.hash_weight).cpu().detach().numpy()
    num_bins = 16
    n, bins, patches = plt.hist(fcw2, num_bins, facecolor="blue", alpha=0.5)
    plt.title("Histogram of fc2 weights")
    plt.savefig("./LeNet_v2/fc2_weight_histogram_%d.pdf" % (VERSION), dpi=1200)
    plt.close()
    n, bins, patches = plt.hist(fhw2, num_bins, facecolor="blue", alpha=0.5)
    plt.title("Histogram of fc2 hashed weights")
    plt.savefig("./LeNet_v2/fc2_hash_weight_histogram_%d.pdf" % (VERSION), dpi=1200)
    plt.close()
