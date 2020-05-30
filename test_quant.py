import torch
import unittest
import torch.nn as nn

from alexnet import AlexNet
from base import LayerData
from quantizer import (
    get_layers_path,
    getattr_by_path_list,
    BitQuantizer,
    setattr_by_path_list,
    use_hashed_conv,
    QuantConv2d,
    quantizeModel,
)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(100, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1000, 10),
        )


class TestAttributeGetAndSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = FakeModel()
        return

    def test_path_list(self):
        layer_paths = get_layers_path(self.model)
        self.assertEqual(
            layer_paths, [[["conv1"]], [["classifier"], 1], [["classifier"], 4],]
        )

    def test_getattr(self):
        layer_paths = [
            [["conv1"]],
            [["classifier"], 1],
            [["classifier"], 4],
        ]
        attrs = []
        for layer_path in layer_paths:
            attr = getattr_by_path_list(self.model, layer_path)
            attrs.append(attr)
        self.assertEqual(
            attrs,
            [self.model.conv1, self.model.classifier[1], self.model.classifier[4],],
        )

    def test_setattr(self):
        model = FakeModel()
        new_linear = nn.Linear(10, 10)

        path_list = [["classifier"], 1]
        setattr_by_path_list(model, path_list, new_linear)

        self.assertEqual(getattr_by_path_list(model, path_list), new_linear)

        path_list = [["conv1"]]
        setattr_by_path_list(model, path_list, new_linear)
        self.assertEqual(getattr_by_path_list(model, path_list), new_linear)


class TestLayerData(unittest.TestCase):
    def test_naming(self):
        qual_path = [["classifier"], 4]
        l = LayerData(qual_path)
        self.assertEqual(l.layer_name, "classifier.4")


class TestQuantizer(unittest.TestCase):
    def test_quantizer(self):
        n_fs = 8
        n_bits = 4
        model = FakeModel()
        avoid = []
        n_iter = 10
        model.cuda()
        bit_quantizer = BitQuantizer(model, n_fs, n_bits, avoid=[], verbose=0)

        bit_quantizer.train_hash_functions(n_iter=n_iter)
        hashed_model = bit_quantizer.get_hashed_model()
        return


@quantizeModel(n_functions=4, n_bits=2)
class FakeQuantModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,)
        self.encoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3,), nn.Conv2d(64, 64, kernel_size=3,)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.encoder(x)
        return x


class TestQuantConv(unittest.TestCase):
    def test_replace_conv(self):
        model = FakeModel()
        model = use_hashed_conv(model)
        self.assertTrue(isinstance(model.conv1, QuantConv2d))

    def test_quant_decorator(self):
        model = FakeQuantModel()
        self.assertTrue(isinstance(model.encoder[0], QuantConv2d))
        self.assertEqual(model.get_hash_loss(), 0)

    def test_hash_loss(self):
        model = FakeQuantModel()
        model.cuda()
        self.assertEqual(float(model.get_hash_loss()), 0)
        x = torch.randn(2, 3, 28, 28).cuda()
        x = model(x)
        self.assertGreater(float(model.get_hash_loss()), 0)


if __name__ == "__main__":
    unittest.main()
