import torch
import torchvision
import torchvision.transforms as transforms

def evaluate(model,loader,cuda):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            if cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            p = model(inputs)
            _,p = torch.max(p.data, 1)

            total += labels.size(0)
            correct += (p == labels).sum().item()

    return correct/total

def cifar10_loader(batch_size=512,data_path="./data"):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                        download=False,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                    download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, num_workers=4)
    return trainloader,testloader

