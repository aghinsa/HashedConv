# Quantized Convolutional layer

Provides two methods to quantize weights of pytorch models.
- quantizeModel : Optimizes hash loss when training
- BitQuantizer : Finds quantized values by minimizing ![formula](https://render.githubusercontent.com/render/math?math=$E[(q(w)-w)^2]$), uses psuedo inverse least square solution.

## Usages

- ### QuantConv2d

```py
from quantizer import quantizeModel,QuantConv2d

@quantizeModel(n_functions = 4,n_bits=2)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, )
        self.encoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, ),
            nn.Conv2d(64, 64, kernel_size=3, )
        )

model = MyModel()
isinstance(model.encoder[0],QuantConv2d)
True
```
- ### BitQuantizer

```py
from quantizer import BitQuantizer

model = MyModel()
n_fs = 32
n_bits = 6
n_iter = 50
avoid =[]
bit_quantizer = BitQuantizer(
            model,
            n_fs,
            n_bits,
            avoid=avoid
                    )
bit_quantizer.train_hash_functions(n_iter = n_iter)
hashed_model = bit_quantizer.get_hashed_model()

```

## Directory Structure

- `alexnet.py` : Modle definition for alexnet and code to train.
- `base.py` : Definitons of classes used in other modules.
- `quantizer` : Main module in which quantization logic is implemented.
- `trainer.py` : Train model to find quantized encodings.
- `grid_search_bit_quantizer` : code to do grid search on hyper parameters and logging for `BitQuantizer`.
- `run_bit_quantizer` : Sample code to hash pretrained model using BitQuantizer.
- `test_quant` : Unintests
    - `python -m unittest test_quant` : to run all tests.
    - `python -m unittest test_quant.ClassName.FunctionName` : to run individual tests.
- `train_and_log_bit_quantizer.py`
- `utils.py` : Helper functions for data loading and model evaluation.


## Legacy Methods

- See README.md in `legacy/$NAME` for more details on each method.

### Linear-Thresholding
- Each forward pass, the weights are distributed based on a linear scale from min-weight to max-weight into the specified number of buckets. The weight corresponding to each bucket is a learnable parameter.
- Experiments done using **LeNet** on **CIFAR10** dataset.

### Polynomial-Hashing
- A 2nd degree polynomial is maintained for each bucket which has a fixed weight. A high-temperature softmax over the function values classifies each weight into a corresponding bucket.
- Experiments are done using **LeNet** on **FashionMNIST** and **CIFAR10** dataset - training from scratch.
- Experiments also done using **ResNet34** on **CIFAR10** - both training from scratch and using pretrained models.

# References
- [CS6886: Systems Engineering for Deep Learning - Prof. Pratyush Kumar, IIT Madras](https://www.cse.iitm.ac.in/~pratyush/CS6886_SysDL.html)
- [ResNet modules](https://github.com/akamaster/pytorch_resnet_cifar10)
- [LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks](https://arxiv.org/abs/1807.10029)
- [A Deep Look into Logarithmic Quantization of ModelParameters in Neural Networks](https://dl.acm.org/doi/pdf/10.1145/3291280.3291800)
- [Bit Efficient Quantization for Deep Neural Networks](https://arxiv.org/pdf/1910.04877.pdf)
