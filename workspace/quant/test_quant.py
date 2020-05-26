import torch
import unittest
import torch.nn as nn

from alexnet import AlexNet
from quantizer import get_layers_path,getattr_by_path_list

class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, )
        self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 2 * 2, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),

		)


class TestAttributeGetter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = FakeModel()
        return

    def test_path_list(self):
        layer_paths = get_layers_path(self.model)
        self.assertEqual(
            layer_paths,
            [
                [["conv1"]],
                [['classifier'], 1],
                [['classifier'], 4],
            ]
        )

    def test_getattr(self):
        layer_paths =   [
                    [["conv1"]],
                    [['classifier'], 1],
                    [['classifier'], 4],
                ]
        attrs = []
        for layer_path in layer_paths:
            attr = getattr_by_path_list(self.model,layer_path)
            attrs.append(attr)
        self.assertEqual(
            attrs,
            [
                self.model.conv1,
                self.model.classifier[1],
                self.model.classifier[4],
            ]
        )




if __name__ == "__main__":
    unittest.main()

