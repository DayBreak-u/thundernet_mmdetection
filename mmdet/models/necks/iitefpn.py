import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16

from ..builder import NECKS


@NECKS.register_module
class LiteFpn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 ):
        super(LiteFpn, self).__init__()

        self.in_channels = in_channels
        # self.conv4 = nn.Conv2d(in_channels[1], out_channels, 1, bias=True)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0], 3, 1, 1, groups=in_channels[0], bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels[0], out_channels, 1, bias=True),
            nn.ReLU(inplace=True),
        )

        # self.conv5 = nn.Conv2d(in_channels[2], out_channels, 1, bias=True)

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[1], in_channels[1], 3, 1, 1, groups=in_channels[1], bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels[1], out_channels, 1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.smmoth1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels[-1], in_channels[-1], 3, 2, 1, groups=in_channels[-1], bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels[-1], out_channels, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        C4_lat = self.conv4(inputs[-2])
        C5_lat = self.conv5(inputs[-1])
        C4_lat = F.interpolate(C5_lat, scale_factor=2, mode="nearest") + C4_lat
        C4_lat = self.smmoth1(C4_lat)

        C6_lat = self.conv6(inputs[-1])
        C7_lat = self.max_pool1(C6_lat)
        C8_lat = self.max_pool2(C7_lat)

        outs = [C4_lat, C5_lat, C6_lat, C7_lat, C8_lat]
        return tuple(outs)
