import torch
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
import torch.utils.model_zoo as model_zoo
from torch.nn.modules.batchnorm import _BatchNorm
from collections import OrderedDict
import torch
from ..builder import BACKBONES
class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]
    #
    # def channel_shuffle(self, x):
    #     batchsize, num_channels, height, width = x.data.size()
    #     # assert (num_channels % 4 == 0)
    #     # x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    #     # x = x.permute(1, 0, 2)
    #     # x = x.reshape(2, -1, num_channels // 2, height, width)
    #     x = x.view(batchsize, 2, num_channels // 2, height, width)
    #     x = torch.transpose(x, 1, 2).contiguous()
    #     x = x.view(batchsize, -1, height, width)
    #     x1, x2 = x.chunk(2, dim=1)
    #     return x1, x2

@BACKBONES.register_module
class ShuffleNetV2(nn.Module):
    def __init__(self, model_size='1.5x',out_indices=( 0,1 , 2, 3),frozen_stages=1 , norm_eval = True):
        super(ShuffleNetV2, self).__init__()
        print('model size is ', model_size)

        self.stage_repeats = [4, 8, 4]
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.model_size = model_size
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.feat_channel = []
        features = []

        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                if i == 0:
                    features.append(ShuffleV2Block(input_channel, output_channel,
                                                        mid_channels=output_channel // 2, ksize=3, stride=2))
                else:
                    features.append(ShuffleV2Block(input_channel // 2, output_channel,
                                                        mid_channels=output_channel // 2, ksize=3, stride=1))

                input_channel = output_channel
            self.__setattr__("feature_%d" % (idxstage + 1), nn.Sequential(*features))
            self.feat_channel.append(input_channel)
            features = []

        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[-1]),
            nn.ReLU(inplace=True)
        )


    def _freeze_stages(self):

        for i in range(self.frozen_stages):
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
                state_dict = torch.load(pretrained, map_location=lambda storage, loc: storage)["state_dict"]
                # print(state_dict.keys())
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

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
        x = self.first_conv(x)
        x = self.maxpool(x)
        y = [x]
        for idxstage in range(len(self.stage_repeats)):
            x = self.__getattr__("feature_%d" % (idxstage+1))(x)
            y.append(x)
        x = self.conv_last(x)
        y[-1] = x
        out = []
        for idx in self.out_indices:
            out.append(y[idx])
        return tuple(out)

    def train(self, mode=True):
        super(ShuffleNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


if __name__ == "__main__":
    model = ShuffleNetV2()
    # print(model)

    test_data = torch.rand(5, 3, 224, 224)
    test_outputs = model(test_data)
    print(test_outputs.size())