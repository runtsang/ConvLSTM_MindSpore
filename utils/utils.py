import mindspore.nn as nn
import mindspore.ops as ops
from collections import OrderedDict

class MaxPool2d(nn.Cell):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.max_pool = ops.MaxPool(kernel_size, stride)
        self.use_pad = padding != 0
        if isinstance(padding, tuple):
            assert len(padding) == 2
            paddings = ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
        elif isinstance(padding, int):
            paddings = ((0, 0),) * 2 + ((padding, padding),) * 2
        else:
            raise ValueError('padding should be a tuple include 2 numbers or a int number')
        self.pad = ops.Pad(paddings)
    
    def construct(self, x):
        if self.use_pad:
            x = self.pad(x)
        return self.max_pool(x)

def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.Conv2dTranspose(in_channels=v[0],
                                                 out_channels=v[1],
                                                 kernel_size=v[2],
                                                 stride=v[3],
                                                 pad_mode='pad',
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU()))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(0.2)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0],
                               out_channels=v[1],
                               kernel_size=v[2],
                               stride=v[3],
                               pad_mode='pad',
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU()))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(0.2)))
        else:
            raise NotImplementedError
    return nn.SequentialCell(OrderedDict(layers))
