# ALEXNET
# source : https://github.com/icpm/pytorch-cifar10/blob/master/models/AlexNet.py

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from utils import evaluate, cifar10_loader

from quantizer import quantizeModel

BATCH_SIZE = 4096
N_EPOCHS = 30
NUM_CLASSES = 10
LOAD_CKPT = None
CUDA = True

if CUDA:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,)
        self.ops1 = nn.Sequential(nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2),)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1,)
        self.ops2 = nn.Sequential(nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2),)

        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1,)
        self.ops3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1,)
        self.ops4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1,)
        self.ops5 = nn.Sequential(nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2),)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ops1(x)
        x = self.conv2(x)
        x = self.ops2(x)
        x = self.conv3(x)
        x = self.ops3(x)
        x = self.conv4(x)
        x = self.ops4(x)
        x = self.conv5(x)
        x = self.ops5(x)
        x = x.reshape(-1, 256 * 2 * 2)
        x = self.classifier(x)
        x = self.softmax(x)
        return x


def get_loss(labels, preds):
    ce = nn.NLLLoss(reduction="mean")
    l1 = ce(torch.log(preds), labels)
    return l1


if __name__ == "__main__":

    trainloader, testloader = cifar10_loader(batch_size=4096, data_path="./data")
    CUDA = True
    model = AlexNet()

    if LOAD_CKPT is not None:
        LOAD_CKPT = f"./checkpoint/alexnet_{LOAD_CKPT}"
        model.load_state_dict(torch.load(LOAD_CKPT))

    if CUDA:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters())
    writer = SummaryWriter()

    global_step = 0

    for epoch in range(1, N_EPOCHS + 1):

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            preds = model(inputs)
            _, logits = torch.max(preds, 1)

            loss = get_loss(labels, preds)

            loss.backward()
            optimizer.step()

            writer.add_scalar("loss/train", loss, global_step)

            global_step += 1
            print(f"loss:{loss}")

        if epoch % 10 == 0:

            model.eval()
            test_acc = evaluate(model, testloader, cuda=CUDA)
            train_acc = evaluate(model, trainloader, cuda=CUDA)

            print(f"Epoch {epoch} complete:")
            print(f"\ttest Accuracy : {test_acc}")
            print(f"\ttrain Accuracy : {train_acc}")
            model.train()

    torch.save(model.state_dict(), f"./model_checkpoints/alexnet.pth")
    print("Finished Training")
