import torch, torchvision
from torch import nn

from utils.box_ops import cat


class SRPooler(nn.Module):
    """
    SRPooler for Detection with or without FPN.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, output_size, scale, sampling_ratio):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(SRPooler, self).__init__()
        self.pooler = torchvision.ops.RoIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
                )
        self.output_size = output_size

    def convert_to_roi_format(self, boxes):
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def forward(self, x, boxes, sr=None):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
            sr(list([BoxList])): search region boxes.
        Returns:
            result (Tensor)
        """
        if sr is None:
            rois = self.convert_to_roi_format(boxes)
        else:
            # extract features for SR when it is none
            rois = self.convert_to_roi_format(sr)
        # import pdb;pdb.set_trace()
        return self.pooler(x, rois)
