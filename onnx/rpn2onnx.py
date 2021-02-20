from shufflenetv2 import ShuffleNetV2
from torch import nn
from torch.functional import F
import torch
from cem import CEM
from iitefpn2 import LiteFpn

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
#
# class LightRPNHead(nn.Module):
#     """RPN head.
#
#     Args:
#         in_channels (int): Number of channels in the input feature map.
#     """  # noqa: W605
#
#     def __init__(self, in_channels, feat_channels):
#         super(LightRPNHead, self).__init__()
#         self.in_channels = in_channels
#         self.feat_channels = feat_channels
#         self.num_anchors = 3
#         self.rpn_conv_dw = nn.Conv2d(
#             self.in_channels, self.in_channels, 5, padding=2, groups=self.in_channels)
#         self.rpn_conv_linear = nn.Conv2d(
#             self.in_channels, self.feat_channels, 1, padding=0)
#
#         self.rpn_cls = nn.Conv2d(self.feat_channels,
#                                  self.num_anchors * 1 , 1)
#         self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)
#         self.sam = nn.Conv2d(
#             self.feat_channels, self.in_channels, 1, padding=0, bias=False)
#         self.bn = nn.BatchNorm2d(self.in_channels)
#         self.act = h_sigmoid()
#
#     def forward(self, x):
#         """Forward feature map of a single scale level."""
#         rpn_out = self.rpn_conv_dw(x)
#         rpn_out = F.relu(rpn_out, inplace=True)
#         rpn_out = self.rpn_conv_linear(rpn_out)
#         rpn_out = F.relu(rpn_out, inplace=True)
#         sam = self.act(self.bn(self.sam(rpn_out)))
#         x = x * sam
#         rpn_cls_score = self.rpn_cls(rpn_out)
#         rpn_bbox_pred = self.rpn_reg(rpn_out)
#         return rpn_cls_score, rpn_bbox_pred, x

class LightRPNHead(nn.Module):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
    """  # noqa: W605

    def __init__(self, in_channels, feat_channels):
        super(LightRPNHead, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_anchors = 15
        self.rpn_conv_exp = nn.Conv2d(
            self.in_channels, self.feat_channels, 1, padding=0)
        self.rpn_conv_dw = nn.Conv2d(
            self.feat_channels, self.feat_channels, 5, padding=2, groups=self.feat_channels)
        self.rpn_conv_linear = nn.Conv2d(
            self.feat_channels, self.feat_channels, 1, padding=0)

        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * 1, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)
        self.sam = nn.Conv2d(
            self.feat_channels, self.in_channels, 1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.act = h_sigmoid()

    def forward(self, x):
        """Forward feature map of a single scale level."""
        rpn_out = self.rpn_conv_exp(x)
        rpn_out = F.relu(rpn_out, inplace=True)
        rpn_out = self.rpn_conv_dw(rpn_out)
        rpn_out = F.relu(rpn_out, inplace=True)
        rpn_out = self.rpn_conv_linear(rpn_out)
        rpn_out = F.relu(rpn_out, inplace=True)
        sam = self.act(self.bn(self.sam(rpn_out)))
        x = x * sam
        rpn_cls_score = self.rpn_cls(rpn_out)
        rpn_bbox_pred = self.rpn_reg(rpn_out)
        return rpn_cls_score, rpn_bbox_pred, x




class ThunderNet(nn.Module):

    def __init__(self):
        super(ThunderNet, self).__init__()
        self.backbone = ShuffleNetV2(frozen_stages=-1,
                                    model_size="1.5x",
                                    norm_eval=False,
                                    out_indices=(2, 3))

        self.neck = CEM(in_channels=[352, 1024],
                        out_channels=216, )
        self.rpn_head = LightRPNHead(216, 512)
    def forward(self, x):
        x = self.backbone(x)
        xs = self.neck(x)
        rpn_cls_score, rpn_bbox_pred, x = self.rpn_head(xs[0])
        # rpn_cls_score = torch.sigmoid(rpn_cls_score)
        return  rpn_cls_score, rpn_bbox_pred, x , xs[1:]


net = ThunderNet()
# state_dict = torch.load(pretrained, map_location=lambda storage, loc: storage)
# state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
pretrained =  "../thundernet_coco_shufflenetv2_1.5/epoch_1.pth"
state_dict = torch.load(pretrained, map_location=lambda storage, loc: storage)["state_dict"]
res = net.load_state_dict(state_dict,strict=False)
print(res)
net.eval()

dummy_input1 = torch.randn(1, 3, 320, 320)
# dummy_input2 = torch.randn(1, 3, 64, 64)
# dummy_input3 = torch.randn(1, 3, 64, 64)
input_names = ["input"]
output_names = ["rpn_cls_score","rpn_bbox_pred","x"]
save_name = "thundernet_shufflenetv2_15_coco_rpn.onnx"
# torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnx", verbose=True, input_names=input_names, output_names=output_names)
torch.onnx.export(net, dummy_input1,save_name , verbose=False, input_names=input_names,
                  output_names=output_names)

import os
os.system("python -m onnxsim {0} {0}".format(save_name))

