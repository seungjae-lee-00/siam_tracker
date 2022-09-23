import torch.nn as nn
import torch.nn.functional as F

from model.track_head.sr_pool import SRPooler


class EMMFeatureExtractor(nn.Module):
    """
    Feature extraction for template and search region is different in this case
    """

    def __init__(self, cfg):
        super(EMMFeatureExtractor, self).__init__()

        resolution = cfg['MODEL']['TRACK_HEAD']['POOLER_RESOLUTION']
        scale = cfg['MODEL']['TRACK_HEAD']['POOLER_SCALES'][0]
        sampling_ratio = cfg['MODEL']['TRACK_HEAD']['POOLER_SAMPLING_RATIO']
        r = cfg['MODEL']['TRACK_HEAD']['SEARCH_REGION']
        
        pooler_z = SRPooler(
            output_size=(resolution, resolution),
            scale=scale,
            sampling_ratio=sampling_ratio)
        pooler_x = SRPooler(
            output_size=(int(resolution*r), int(resolution*r)),
            scale=scale,
            sampling_ratio=sampling_ratio)

        self.pooler_x = pooler_x
        self.pooler_z = pooler_z

    def forward(self, x, proposals, sr=None):
        if sr is not None:
            x = self.pooler_x(x, proposals, sr)
        else:
            x = self.pooler_z(x, proposals)

        return x


class EMMPredictor(nn.Module):
    def __init__(self, cfg):
        super(EMMPredictor, self).__init__()

        if cfg['MODEL']['BACKBONE']['CONV_BODY'].startswith("DLA"):
            in_channels = cfg['MODEL']['DLA']['BACKBONE_OUT_CHANNELS']

        self.cls_tower = make_conv3x3(in_channels=in_channels, out_channels=in_channels,
                                      use_relu=True, kaiming_init=False)
        self.reg_tower = make_conv3x3(in_channels=in_channels, out_channels=in_channels,
                                      use_relu=True, kaiming_init=False)
        self.cls = make_conv3x3(in_channels=in_channels, out_channels=2, kaiming_init=False)
        self.center = make_conv3x3(in_channels=in_channels, out_channels=1, kaiming_init=False)
        self.reg = make_conv3x3(in_channels=in_channels, out_channels=4, kaiming_init=False)

    def forward(self, x):
        cls_x = self.cls_tower(x)
        reg_x = self.reg_tower(x)
        cls_logits = self.cls(cls_x)
        center_logits = self.center(cls_x)
        reg_logits = F.relu(self.reg(reg_x))

        return cls_logits, center_logits, reg_logits


# class EMMPredictor(nn.Module):
#     def __init__(self, cfg):
#         super(EMMPredictor, self).__init__()

#         if cfg['MODEL']['BACKBONE']['CONV_BODY'].startswith("DLA"):
#             in_channels = cfg['MODEL']['DLA']['BACKBONE_OUT_CHANNELS']

#         self.cls_tower_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
#         nn.init.xavier_normal_(self.cls_tower_conv.weight)
#         self.cls_tower_relu = nn.ReLU(inplace=True)
#         self.reg_tower_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
#         nn.init.xavier_normal_(self.reg_tower_conv.weight)
#         self.reg_tower_relu = nn.ReLU(inplace=True)
#         self.cls = nn.Conv2d(in_channels, 2, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
#         nn.init.xavier_normal_(self.cls.weight)
#         self.center = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
#         nn.init.xavier_normal_(self.center.weight)
#         self.reg = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
#         self.reg_relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         cls_x = self.cls_tower_conv(x)
#         cls_x_relu = self.cls_tower_relu(cls_x)
#         reg_x = self.reg_tower_conv(x)
#         reg_x_relu = self.reg_tower_relu(reg_x)
#         cls_logits = self.cls(cls_x_relu)
#         center_logits = self.center(cls_x_relu)
#         reg_logits = self.reg(reg_x_relu)
#         reg_logits = self.reg_relu(reg_logits)

#         return cls_logits, center_logits, reg_logits


def make_conv3x3(
    in_channels,
    out_channels,
    dilation=1,
    stride=1,
    use_relu=False,
    kaiming_init=True
):
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=True
    )
    if kaiming_init:
        nn.init.kaiming_normal_(
            conv.weight, mode="fan_out", nonlinearity="relu"
        )
    else:
        nn.init.normal_(conv.weight, std=0.01)
    module = [conv,]
    if use_relu:
        module.append(nn.ReLU(inplace=True))
    if len(module) > 1:
        return nn.Sequential(*module)
    return conv