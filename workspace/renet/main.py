import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from alexnet import AlexNet

import torch.optim as optim

BATCH_SIZE = 512

LOAD_CKPT = None

def evaluate(model,loader):
    correct = [0,0]
    total = 0
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()
            p,ph,_,_ = model(inputs)
            _,p = torch.max(p.data, 1)
            _,ph = torch.max(ph.data, 1)

            total += labels.size(0)
            correct[0] += (p == labels).sum().item()
            correct[1] += (ph == labels).sum().item()

    return (correct[0]/total,correct[1]/total)

def hash_loss(labels,logits,logits_hash,ps,qs):
    kl = nn.KLDivLoss(reduction="batchmean")
    ce = nn.NLLLoss(reduction="mean")


    l1 = ce(torch.log(logits),labels)

    # print(f"l1:{l1}")
    kl_loss1 = 0
    for p,q in zip(ps,qs):
        #kl expects target as prob and inputs as log prob
        q=torch.log(q)
        kl_loss1 += kl(q,p)
        # print(f"kl:{kl_loss1}")

    # print(logits_hash[0])
    # print(logits[0])

    kl_loss2 = kl(torch.log(logits_hash),logits)
    # print(f"kl2:{kl_loss2}")

    # print()

    loss = l1+kl_loss1+kl_loss2

    return loss

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def get_accuracy(labels,logits,reduce_mean):
    acc = (labels == logits).sum()
    if reduce_mean:
        acc = acc/logits.size()[0]
    return acc
if __name__ == "__main__":
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    download_data = False
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=download_data,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=download_data, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



    CUDA =True

    model = AlexNet()

    if LOAD_CKPT is not None:
        LOAD_CKPT = f"./checkpoint/model_{LOAD_CKPT}"
        model.load_state_dict(torch.load(LOAD_CKPT))

    if CUDA:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
    writer = SummaryWriter()

    N_EPOCHS = 5
    global_step = 0

    for epoch in range(1,N_EPOCHS+1):

        running_loss = 0.0
        running_acc = 0.0
        running_acc_hash = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()


            optimizer.zero_grad()

            preds,preds_hash,ps,qs = model(inputs)
            _,logits = torch.max(preds,1)
            _,logits_hash = torch.max(preds_hash,1)


            loss = hash_loss(labels,preds,preds_hash,ps,qs)
            loss.backward()
            optimizer.step()

            acc = get_accuracy(labels,logits,reduce_mean = True)
            acc_hash = get_accuracy(labels,logits_hash,reduce_mean = True)

            n=i*BATCH_SIZE
            running_loss += loss.item()
            running_acc = (running_acc*n + (acc*BATCH_SIZE) )/(n+1)
            running_acc_hash = (running_acc_hash*n + acc_hash*BATCH_SIZE )/(n+1)


            writer.add_scalar('Loss/train', loss , global_step)
            writer.add_scalar('Accuracy/train_original', acc , global_step)
            writer.add_scalar('Accuracy/train_hashed', acc_hash , global_step)

            global_step+=1

            print(f"loss:{loss}")

            if i % 1000 == 0:
                print(f"Loss : {running_loss}")
                print(f"Accuracy : {running_acc}")
                print(f"Accuracy_hash : {running_acc_hash}")

                running_loss = 0.0
                running_acc = 0.0
                running_acc_hash = 0.0


        torch.save(model.state_dict(), f"./checkpoint/model_{epoch}")

        if epoch%10 == 0:
            model.eval()
            test_acc,test_acc_h = evaluate(model,testloader)
            train_acc,train_acc_h = evaluate(model,trainloader)

            print(f"Epoch {epoch} complete:")
            print(f"\tloss : {running_loss}")
            print(f"\ttest Accuracy : {test_acc}")
            print(f"\ttest Accuracy_hash : {test_acc_h}")
            print(f"\ttrain Accuracy : {train_acc}")
            print(f"\ttrain Accuracy_hash : {train_acc_h}")
            model.train()
    print('Finished Training')