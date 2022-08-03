import torch
import torch.nn as nn

from model.rpn.rpn import build_rpn
from model.box_head.roi import build_roi_box_head
import model.backbone.backbone as backbone

class FasterRCNN(nn.Module):
    def __init__(self, cfg, training=True, track_training=False):
        super(FasterRCNN, self).__init__()

        self.track_training = track_training
        self.backbone = backbone.dla(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels, self.track_training)
        self.roi_head = build_roi_box_head(cfg, self.backbone.out_channels)
        self.training = training

    def forward(self, images, targets=None, given_detection=None):

        # if self.training and targets is None:
        #     raise ValueError("In training mode, targets should be passed")
        images = [image.unsqueeze(0) for image in images]
        images = torch.vstack(images)
        images = images.cuda()
        features = self.backbone(images)
        proposals, proposal_losses = self.rpn(images, features, targets)

        if self.roi_head:
            x, result, roi_losses = self.roi_head(features,
                                                   proposals,
                                                   targets
                                                   )

        else:
            raise NotImplementedError

        if self.training and not self.track_training:
            losses = {}
            losses.update(roi_losses)
            losses.update(proposal_losses)
            return features, proposals, result, losses

        if self.track_training:
            losses = {}
            losses.update(roi_losses)
            losses.update(proposal_losses)
            return features, proposals, result, losses, images

        return features, proposals, result


# def build_siammot(cfg):
#     siammot = SiamMOT(cfg)
#     return siammot
