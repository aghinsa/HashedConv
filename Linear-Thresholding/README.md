# Linear-Thresholding
- Each forward pass, the weights are distributed based on a linear scale from min-weight to max-weight into the specified number of buckets. The weight corresponding to each bucket is a learnable parameter.
- The file 'Hashed.py' contains the *Hashed* version of the 'Conv2d' module.
- The file 'LeNet_CIFAR10_v1.py' has the model spec and training code for LeNet architecture on the CIFAR10 dataset. Only the 'Conv2d' layers have been hashed. Model is being trained from scratch.