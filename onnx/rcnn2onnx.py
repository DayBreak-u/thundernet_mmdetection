from mobilenetv2 import MobileNetV2
from torch import nn
from torch.functional import F
import torch
from cem import CEM


class Bbox_head(nn.Module):

    def __init__(self):
        super(Bbox_head, self).__init__()

        self.shared_fcs = nn.ModuleList( )
        self.shared_fcs.append(nn.Linear(245, 1024))
        self.shared_fcs.append(nn.Linear(1024, 1024))

        self.fc_reg = nn.Linear(1024, 20 * 4)
        self.fc_cls = nn.Linear(1024, 21)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for fc in self.shared_fcs:
            x = F.relu(fc(x))

        cls_score = self.fc_cls(x)
        print(cls_score)
        cls_score = F.softmax(cls_score)
        print(cls_score)
        bbox_pred = self.fc_reg(x)

        return cls_score, bbox_pred


class ROIHead(nn.Module):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
    """  # noqa: W605

    def __init__(self):
        super(ROIHead, self).__init__()
        self.bbox_head = Bbox_head()

    def forward(self, x):
        cls_score, bbox_pred = self.bbox_head(x)
        return cls_score, bbox_pred


class ThunderNet(nn.Module):

    def __init__(self):
        super(ThunderNet, self).__init__()

        self.roi_head = ROIHead()

    def forward(self, x):
        cls_score, bbox_pred = self.roi_head(x)
        return cls_score, bbox_pred


net = ThunderNet()
print(net.state_dict().keys())

pretrained = "./epoch_21.pth"
state_dict = torch.load(pretrained, map_location=lambda storage, loc: storage)["state_dict"]
# print(state_dict.keys())

res = net.load_state_dict(state_dict, strict=False)
print(res)
net.eval()

dummy_input1 = torch.randn(1, 5, 7, 7)
# dummy_input2 = torch.randn(1, 3, 64, 64)
# dummy_input3 = torch.randn(1, 3, 64, 64)
input_names = ["roi_feat"]
output_names = ["cls_score", "bbox_pred"]
# torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnx", verbose=True, input_names=input_names, output_names=output_names)
torch.onnx.export(net, dummy_input1, "thundernet_mbv2_rcnn.onnx", verbose=False, input_names=input_names,
                  output_names=output_names)
