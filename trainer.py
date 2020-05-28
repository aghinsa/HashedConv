import torch
import numpy as np
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from alexnet import AlexNet
from pytorch_resnet.resnet import resnet32

from utils import evaluate,cifar10_loader
from quantizer import quantizeModel

def get_loss(labels,preds,model):
    ce = nn.CrossEntropyLoss(reduction="mean")
    l1 = ce(preds,labels)

    return l1



if __name__ == "__main__":
    batch_size = 256
    n_epochs = 100

    trainloader,testloader = cifar10_loader(batch_size = batch_size,data_path="../data")

    model = resnet32()
    model_name = "resnet_quantized"
    model = quantizeModel(n_bits=6,n_functions=64)(model)
    model.cuda()


    optimizer = torch.optim.Adam(model.parameters())
    writer = SummaryWriter()

    global_step = 0


    for epoch in range(1,n_epochs+1):

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            inputs = inputs.cuda()
            labels = labels.cuda()


            optimizer.zero_grad()

            preds = model(inputs)
            _,logits = torch.max(preds,1)

            loss = get_loss(labels,preds,model) + model.get_hash_loss()
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss/train', loss , global_step)

            global_step+=1
            print(f"loss:{loss}")


        if epoch%1==0:
            # torch.save(model.state_dict(), f"./checkpoint/model_{epoch}")
            trl,tsl = cifar10_loader(batch_size=4096,data_path="../data")
            normal_test_acc = evaluate(model,tsl,cuda = True)
            normal_train_acc = evaluate(model,trl,cuda = True)

            print(f"Epoch {epoch} complete:")
            print(f"\ttest Accuracy : { normal_test_acc}")
            print(f"\ttrain Accuracy : {normal_train_acc}")

            model.eval()
            hashed_test_acc = evaluate(model,tsl,cuda = True)
            hashed_train_acc = evaluate(model,trl,cuda = True)
            print(f"\teval_test Accuracy : { hashed_test_acc}")
            print(f"\teval_train Accuracy : {hashed_train_acc}")
            model.train()


            writer.add_scalars('train_accuracy',
                {
                    'normal': normal_train_acc,
                    'hashed': hashed_train_acc
                },
                epoch
            )

            writer.add_scalars('test_accuracy',
                {
                    'normal': normal_test_acc,
                    'hashed': hashed_test_acc
                },
                epoch
            )



    model.copy_hashed_weights()
    torch.save(model.state_dict(), f"./checkpoint/{model_name}")
    print('Finished Training')

