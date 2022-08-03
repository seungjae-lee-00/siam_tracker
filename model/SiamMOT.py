import torch
import torch.nn as nn
import torchvision

from model.track_head.track import build_track_head
from model.solver.solver import builder_tracker_solver
from utils.track_utils import build_track_utils
from utils.box_ops import cat_boxlist
from utils.bbox import BoxList
import utils.plot_box
import numpy as np
import cv2
import os, sys

import time

mean = np.asarray([ 0.485, 0.456, 0.406 ])
std = np.asarray([ 0.229, 0.224, 0.225 ])


class SiamMOT(nn.Module):
    def __init__(self, cfg, detector):
        super(SiamMOT, self).__init__()

        self.cfg = cfg
        self.detector = detector.train()

        track_utils, track_pool = build_track_utils(cfg)
        self.trackhead = build_track_head(cfg, track_utils, track_pool)
        self.solver = builder_tracker_solver(cfg, track_pool)
        self.track_memory = None
        self.idx = 0

    def flush_memory(self, tracks_in_memory=None):
        self.track_memory = tracks_in_memory

    def reset_siammot_status(self):
        self.flush_memory()
        self.roi_heads.reset_roi_status()

    def forward(self, images, targets=None, given_detection=None, out_dir=None):
        if self.training:
            with torch.no_grad():
                features, proposals, result, loss_detector = self.detector(images, targets)
            # features, proposals, result, loss_detector = self.detector(images, targets)
        else :
            end_time = time.time()
            features, proposals, result = self.detector(images, targets)
            detector_time = time.time() - end_time

        end_time = time.time()
        y, tracks, loss_track = self.trackhead(features, 
                                               proposals, 
                                               targets, 
                                               self.track_memory
                                               )
        tracker_time = time.time() - end_time

        if not self.training:
            end_time = time.time()

            if tracks is not None:
                tracks_refined =self._refine_tracks(features, tracks)
                result = [cat_boxlist(result+tracks_refined)]

            detections = self.solver(result)
            tracks_in_memory = self.trackhead.get_track_memory(features, detections)
            self.flush_memory(tracks_in_memory=tracks_in_memory)

        if self.training:
            losses = {}
            losses.update(loss_track)
            # losses.update(loss_detector)
            return result, losses

        if out_dir is not None:
            out_name = os.path.join(out_dir, 'track_test_{0:06d}'.format(self.idx))
        else :
            out_name = 'track_test_{0:06d}'.format(self.idx)
        if tracks is not None:
            utils.plot_box.boxes_on_image(images, detections, out_name)
        
        self.idx += 1

        return tracks

    def _refine_tracks(self, features, tracks):
        """
        Use box head to refine the bounding box location
        The final vis score is an average between appearance and matching score
        """
        if len(tracks[0]) == 0:
            return tracks[0]
        track_scores = tracks[0].get_field('scores') + 1.
        _, tracks, _ = self.detector.roi_head(features, tracks)
        det_scores = tracks[0].get_field('scores')
        det_boxes = tracks[0].bbox

        if self.cfg['MODEL']['TRACK_HEAD']['TRACKTOR']:
            scores = det_scores
        else:
            scores = (det_scores + track_scores) / 2.
        boxes = det_boxes

        r_tracks = BoxList(boxes, image_size=tracks[0].size, mode=tracks[0].mode)
        r_tracks.add_field('scores', scores)
        r_tracks.add_field('ids', tracks[0].get_field('ids'))
        r_tracks.add_field('labels', tracks[0].get_field('labels'))

        return [r_tracks]