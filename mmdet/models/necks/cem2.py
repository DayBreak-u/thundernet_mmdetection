import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from ..builder import NECKS

@NECKS.register_module
class CEM2(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                            ):
        super(CEM2, self).__init__()

        self.in_channels = in_channels
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0], 3, 1, 1, groups=in_channels[0], bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels[0], out_channels, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[1], in_channels[1], 3, 1, 1, groups=in_channels[1], bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels[1], out_channels, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.smmoth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.convlast = nn.Conv2d(in_channels[1], out_channels, 1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.init_weights()
    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        C4_lat = self.conv4(inputs[-2])
        C5_lat = self.conv5(inputs[-1])
        C4_lat = F.interpolate(C5_lat,scale_factor=2,mode="nearest") + C4_lat

        avg_pool = self.avg_pool(inputs[-1])
        C4_lat = F.relu(self.convlast(avg_pool)) + C4_lat
        C4_lat = self.smmoth(C4_lat)


        return tuple([C4_lat])
