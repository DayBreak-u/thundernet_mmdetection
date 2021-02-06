import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)

from torch.nn.modules.batchnorm import _BatchNorm
from collections import OrderedDict
import torch
from ..builder import BACKBONES
import torch.utils.model_zoo as model_zoo

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

@BACKBONES.register_module
class MobileNetV2(nn.Module):
    def __init__(self,width_mult=1.0,round_nearest=8,out_indices=( 0,1 , 2, 3),frozen_stages=1 , norm_eval = True):
        super(MobileNetV2, self).__init__()
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages
        block = InvertedResidual
        input_channel = 32
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1], # 0
            [6, 24, 2, 2], # 1
            [6, 32, 3, 2], # 2
            [6, 64, 4, 2], # 3
            [6, 96, 3, 1], # 4
            [6, 160, 3, 2],# 5
            [6, 320, 1, 1],# 6
        ]
        self.feat_id = [1,2,4,6]
        self.feat_channel = []
        self.out_indices = out_indices

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]

        # building inverted residual blocks
        for id,(t, c, n, s) in enumerate(inverted_residual_setting):
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
            if id in self.feat_id  :
                self.__setattr__("feature_%d"%id,nn.Sequential(*features))
                self.feat_channel.append(output_channel)
                features = []

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _freeze_stages(self):

        for i in range( self.frozen_stages ):
            m = getattr(self, 'feature_{}'.format(self.feat_id[i]))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def load_model(self, state_dict):
        new_model = self.state_dict()
        new_keys = list(new_model.keys())
        old_keys = list(state_dict.keys())
        restore_dict = OrderedDict()
        for id in range(len(new_keys)):
            restore_dict[new_keys[id]] = state_dict[old_keys[id]]
        res = self.load_state_dict(restore_dict)
        print(res)

    def init_weights(self, pretrained=None):
        # from collections import OrderedDict
        # temp = OrderedDict()

        if isinstance(pretrained, str):
            if pretrained.startswith("https"):
                state_dict = model_zoo.load_url(pretrained,
                                                progress=True)
            else:
                state_dict = torch.load(pretrained,map_location=lambda  storage,loc:storage)
                state_dict = {k.replace("module.",""):v for k,v in state_dict.items() }

            self.load_model(state_dict)

        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        y = []
        for id in self.feat_id:
            x = self.__getattr__("feature_%d"%id)(x)
            y.append(x)
        out = []
        for idx in self.out_indices:
            out.append(y[idx])
        return tuple(out)

    def train(self, mode=True):
        super(MobileNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()