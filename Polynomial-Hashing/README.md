# Polynomial-Hashing

- A 2nd degree polynomial is maintained for each bucket which has a fixed weight. A high-temperature softmax over the function values classifies each weight into a corresponding bucket.
- Train loss is a weighted sum of the original network loss and the hashed network loss.
- The file `Hashed.py` contains the **_Hashed_** version of the `Conv2d` and `Linear` modules.
- The file `**LeNet**_**FashionMNIST**_v1.py` has the model spec and training code for **LeNet** architecture on the **FashionMNIST** dataset. Only the `Conv2d` layers have been hashed. Model is being trained from scratch.
- The file `**LeNet**_**CIFAR10**_v2.py` has the model spec and training code for **LeNet** architecture on the **CIFAR10** dataset. Only the `Conv2d` layers have been hashed. Model is being trained from scratch.
- The file `**LeNet**_**CIFAR10**_v2.py` has the model spec and training code for **LeNet** architecture on the **CIFAR10** dataset. Both the `Conv2d` and `Linear` layers have been hashed except the last layer. Model is being trained from scratch.
- The file `resnet.py` has the model spec for the hashed versions of all **ResNet** architectures. 
- The file `ResNet_**CIFAR10**_v1.py` has the training code for **ResNet34** architecture on the **CIFAR10** dataset. Only the `Conv2d` layers have been hashed. Model is being trained from scratch.
- The file `ResNet_**CIFAR10**_v2.py` has the training code for **ResNet34** architecture on the **CIFAR10** dataset. Only the `Conv2d` layers have been hashed. A pretrained model is imported and only the hashing parameters are being trained.

