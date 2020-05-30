# Linear-Thresholding
- Each forward pass, the weights are distributed based on a linear scale from min-weight to max-weight into the specified number of buckets. The weight corresponding to each bucket is a learnable parameter.
- Train loss is a weighted sum of the original network loss, the hashed network loss and the mean squared error between the output of each hashed layer over the original input.
- The file `Hashed.py` contains the **_Hashed_** version of the `Conv2d` module.
- The file `LeNet_CIFAR10_v1.py` has the model spec and training code for **LeNet** architecture on the **CIFAR10** dataset. Only the `Conv2d` layers have been hashed. Model is being trained from scratch.