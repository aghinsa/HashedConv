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

# References
- [LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks](https://arxiv.org/abs/1807.10029)
- [A Deep Look into Logarithmic Quantization of ModelParameters in Neural Networks](https://dl.acm.org/doi/pdf/10.1145/3291280.3291800)
- [Bit Efficient Quantization for Deep Neural Networks](https://arxiv.org/pdf/1910.04877.pdf)