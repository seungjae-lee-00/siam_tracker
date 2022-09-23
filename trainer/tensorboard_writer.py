import torch
import itertools
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class TensorboardWriter(SummaryWriter):
    def __init__(self, cfg, train_dir):
        super(TensorboardWriter, self).__init__(log_dir=train_dir)

        device = torch.device('cuda')
        self.model_mean = torch.as_tensor(cfg['INPUT']['PIXEL_MEAN'], device=device)
        self.model_std = torch.as_tensor(cfg['INPUT']['PIXEL_STD'], device=device)

        # self.image_to_bgr255 = cfg.INPUT.TO_BGR255

        # # number of images per row during visualization
        # self.num_col = cfg.VIDEO.RANDOM_FRAMES_PER_CLIP
        self.image_to_bgr255 = False
        self.num_col = 2

    def __call__(self, iter, loss, loss_dict, targets):
        """

        :param iter:
        :param loss_dict:
        :param images:
        :return:
        """
        self.add_scalar('loss', loss.detach().cpu().numpy(), iter)
        for (_loss_key, _val) in loss_dict.items():
            self.add_scalar(_loss_key, _val.detach().cpu().numpy(), iter)

    def images_with_boxes(self, images, boxes):
        """
        Get images inpainted with bounding boxes
        :param images: A batch of images are packed in a torch tensor BxCxHxW
        :param boxes:  A list of bounding boxes for the corresponding images
        :param ncols:
        """
        # To numpy array
        images = images.detach().cpu().numpy()
        # new stitched image
        batch, channels, height, width = images.shape
        assert batch % self.num_col == 0
        nrows = batch // self.num_col

        new_height = height * nrows
        new_width = width * self.num_col

        merged_image = np.zeros([channels, new_height, new_width])
        bbox_in_merged_image = []

        for img_idx in range(batch):
            row = img_idx // self.num_col
            col = img_idx % self.num_col
            merged_image[:, row * height:(row + 1) * height, col * width:(col + 1) * width] = \
                images[img_idx, :, :, :]
            box = boxes[img_idx].bbox.detach().cpu().numpy()
            if box.size > 0:
                box[:, 0] += col * width
                box[:, 1] += row * height
                box[:, 2] += col * width
                box[:, 3] += row * height
                bbox_in_merged_image.append(box)

        bbox_in_merged_image = np.array(list(itertools.chain(*bbox_in_merged_image)))

        return merged_image, bbox_in_merged_image
