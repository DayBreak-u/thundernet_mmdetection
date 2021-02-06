import torch.nn as nn
import torch.nn.functional as F


class CEM(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                            ):
        super(CEM, self).__init__()

        self.in_channels = in_channels
        self.conv4 = nn.Conv2d(in_channels[0], out_channels, 1, bias=True)
        self.conv5 = nn.Conv2d(in_channels[1], out_channels, 1, bias=True)
        self.convlast = nn.Conv2d(in_channels[1], out_channels, 1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)


    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        C4_lat = F.relu(self.conv4(inputs[-2]))
        C5_lat = F.relu(self.conv5(inputs[-1]))
        C5_lat = F.interpolate(C5_lat,scale_factor=2,mode="nearest")
        avg_pool = self.avg_pool(inputs[-1])
        Cglb_lat = F.relu(self.convlast(avg_pool))

        outs = [C4_lat + C5_lat + Cglb_lat]


        return outs[0]
