import os, json, yaml
import pickle

import itertools
import torch
import torch.nn.functional as F
import torchvision

import numpy as np
from random import shuffle
from PIL import Image, ImageDraw
import pandas as pd
import xml.etree.ElementTree as elemTree

from utils.bbox import BoxList
# from utils.bbox import BoxList
from model.transforms import build_transforms

import utils.plot_box
import cv2


def verify_data(img_name, out_name, target):
    img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (1152, 768))
    for idx, bbox in enumerate(target.bbox.tolist()):
        track_id = target.get_field('ids')[idx]
        utils.plot_box.plot_one_box(bbox, img, label="track_id : {}".format(track_id), line_thickness=1)
        # utils.plot_box.plot_one_box(bbox, img, label="track_id : {}".format('proposal_box'), line_thickness=1)
    cv2.imwrite(out_name, img)

def draw_boxes(image, point1, point2, color=(0,0,255)):
    draw = ImageDraw.Draw(image)
    draw.rectangle((point1, point2), outline=color, width=2)
    return image

class FeatureMapDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, is_train=True):
        csv_file = cfg['DATASETS']['TRAIN'] if is_train else cfg['DATASETS']['TEST']
        self.data_root = cfg['DATASETS']['ROOT_DIR']
        self.df = pd.read_csv(os.path.join(self.data_root, csv_file))
        self.device = torch.device(cfg['MODEL']['DEVICE'])

        self.class_index = {'PEDESTRIAN' : 1,
                            'VEHICLE' : 2,
                            'CYCLIST' : 3,
                            }

    def __getitem__(self, idx):
        img1_name = self.df.img_1[idx]
        img2_name = self.df.img_2[idx]

        ann1 = img1_name.replace('.jpg','.json')
        ann2 = img2_name.replace('.jpg','.json')

        f_data1 =os.path.join(self.data_root, 'M578_detected/', img1_name).replace("jpg","pkl")
        f_data2 =os.path.join(self.data_root, 'M578_detected/', img2_name).replace("jpg","pkl")

        feature_map1, proposal1 = self.get_pkl_info(f_data1)
        feature_map2, proposal2 = self.get_pkl_info(f_data2)
        
        target1, obj_ids1 = self.get_ann_info((1920,1280), ann1)
        target2, obj_ids2 = self.get_ann_info((1920,1280), ann2)

        obj_ids1, obj_ids2 = self.id_pairing(obj_ids1, obj_ids2)

        target1.add_field('ids', torch.as_tensor(obj_ids1, dtype=torch.int64))
        target2.add_field('ids', torch.as_tensor(obj_ids2, dtype=torch.int64))

        target1 = target1.resize((1152, 768))
        target2 = target2.resize((1152, 768))

        # verified_img1 = os.path.join(self.data_root, 'selectedJPEG_M578/', img1_name)
        # verified_img2 = os.path.join(self.data_root, 'selectedJPEG_M578/', img2_name)

        # verify_data(verified_img1, "test1.jpg", proposal1)
        # verify_data(verified_img2, "test2.jpg", proposal2)
        # verify_data(verified_img1, "test1.jpg", target1)
        # verify_data(verified_img2, "test2.jpg", target2)
        return [[feature_map1, feature_map2], [proposal1, proposal2], [target1, target2]]


    def __len__(self):
        return len(self.df)

    def get_pkl_info(self, pkl_file):
        with open(pkl_file, 'rb') as f:
            u_data = pickle._Unpickler(f)
            u_data.encoding = "latin1"
            data = u_data.load()
            feature_map = np.squeeze(data["feature_map"])
            feature_map = torch.from_numpy(feature_map).cuda()
            boxes = data["proposals"][:,1:5].tolist()
            boxes = torch.as_tensor(boxes, device=self.device, dtype=torch.float).reshape(-1,4)

            proposals = BoxList(boxes, (1152, 768), mode="xyxy")            
            objectness = torch.from_numpy(data["objectness"])
            proposals.add_field("objectness", objectness)
            return feature_map, proposals
    
    def get_ann_info(self, img_size, ann_name):
        ann_path = os.path.join(self.data_root,'Annotations/', ann_name)
        with open(ann_path, 'r') as json_file:
            json_data = json.load(json_file)

        objects = json_data['objects']
        boxes = []
        cls_ids = []
        obj_ids = []

        for obj in objects:
            if (obj['class'] != 'tl') and (obj['class'] != 'ts'):
                cls_id = torch.tensor(self.class_index[obj['class']], dtype=torch.int64)
                bbox = obj['xyxy']
                obj_id = obj['object_id']
                boxes.append(bbox)
                cls_ids.append(cls_id)
                obj_ids.append(obj_id)

        boxes = torch.as_tensor(boxes, device=self.device).reshape(-1,4)
        target = BoxList(boxes, img_size, mode="xyxy")
        target.add_field("labels", torch.as_tensor(cls_ids, dtype=torch.int64))
        return target, obj_ids

    def id_pairing(self, obj_ids1, obj_ids2):
        paired_id1 = list(range(1,len(obj_ids1)+1))        
        paired_id2 = []
        try:
            new_id = paired_id1[-1]+1
        except:
            import pdb;pdb.set_trace()

        for obj in obj_ids2:
            if obj in obj_ids1:
                matched_id = paired_id1[obj_ids1.index(obj)]
                paired_id2.append(matched_id)
            else :
                paired_id2.append(new_id)
                new_id = new_id+1
        return paired_id1, paired_id2


class SequenceImageDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transforms=None, is_train=True):
        csv_file = cfg['DATASETS']['TRAIN'] if is_train else cfg['DATASETS']['TEST']
        self.data_root = cfg['DATASETS']['ROOT_DIR']
        self.df = pd.read_csv(os.path.join(self.data_root, csv_file))
        self.transforms = transforms
        self.device = torch.device(cfg['MODEL']['DEVICE'])

        self.class_index = {'PEDESTRIAN' : 1,
                            'VEHICLE' : 2,
                            'CYCLIST' : 3,
                            }

    def __getitem__(self, idx):
        img1_name = self.df.img_1[idx]
        img2_name = self.df.img_2[idx]

        ann1 = img1_name.replace('.jpg','.json')
        ann2 = img2_name.replace('.jpg','.json')

        img1 = Image.open(os.path.join(self.data_root, 'JPEGImage/', img1_name)).convert("RGB")
        img2 = Image.open(os.path.join(self.data_root, 'JPEGImage/', img2_name)).convert("RGB")

        target1, obj_ids1 = self.get_ann_info(img1, ann1)
        target2, obj_ids2 = self.get_ann_info(img2, ann2)

        obj_ids1, obj_ids2 = self.id_pairing(obj_ids1, obj_ids2)

        target1.add_field('ids', torch.as_tensor(obj_ids1, dtype=torch.int64))
        target2.add_field('ids', torch.as_tensor(obj_ids2, dtype=torch.int64))

        if self.transforms is not None:
            img1, target1 = self.transforms(img1, target1)
            img2, target2 = self.transforms(img2, target2)
        
        return [[img1, img2], [target1, target2]]

    def __len__(self):
        return len(self.df)

    def get_ann_info(self, img, ann_name):
        ann_path = os.path.join(self.data_root,'Annotations/', ann_name)
        with open(ann_path, 'r') as json_file:
            json_data = json.load(json_file)

        objects = json_data['objects']
        boxes = []
        cls_ids = []
        obj_ids = []

        for obj in objects:
            if (obj['class'] != 'tl') and (obj['class'] != 'ts'):
                cls_id = torch.tensor(self.class_index[obj['class']], dtype=torch.int64)
                bbox = obj['xyxy']
                obj_id = obj['object_id']
                boxes.append(bbox)
                cls_ids.append(cls_id)
                obj_ids.append(obj_id)

        boxes = torch.as_tensor(boxes, device=self.device).reshape(-1,4)
        target = BoxList(boxes, img.size, mode="xyxy")
        target.add_field("labels", torch.as_tensor(cls_ids, dtype=torch.int64))

        return target, obj_ids

    def id_pairing(self, obj_ids1, obj_ids2):
        paired_id1 = list(range(1,len(obj_ids1)+1))        
        paired_id2 = []
        try:
            new_id = paired_id1[-1]+1
        except:
            import pdb;pdb.set_trace()

        for obj in obj_ids2:
            if obj in obj_ids1:
                matched_id = paired_id1[obj_ids1.index(obj)]
                paired_id2.append(matched_id)
            else :
                paired_id2.append(new_id)
                new_id = new_id+1
        return paired_id1, paired_id2


def sequence_image_collate(batch):
    images = []
    targets = []

    for data in batch:
        images.append(data[0][0])
        images.append(data[0][1])
        targets.append(data[1][0])
        targets.append(data[1][1])
        
    return images, targets

def sequence_featuremap_collate(batch):

    feature_maps = []
    proposals = []
    targets = []

    for data in batch:
        feature_maps.append(data[0][0])
        feature_maps.append(data[0][1])
        proposals.append(data[1][0])
        proposals.append(data[1][1])
        targets.append(data[2][0])
        targets.append(data[2][1])
        
    return feature_maps, proposals, targets


def build_featuremap_dataloader(cfg, is_train=True):
    batch_size = cfg['DATALOADER']['BATCH_SIZE']
    dataset = FeatureMapDataset(cfg, is_train=is_train)
    data_loader = torch.utils.data.DataLoader(dataset, 
                                              num_workers=cfg['DATALOADER']['NUM_WORKERS'],
                                              batch_size=batch_size,
                                              shuffle=True,
                                              collate_fn=sequence_featuremap_collate)
                                              
    return data_loader


def build_featuremap_dataset(cfg, is_train=True):
    batch_size = cfg['DATALOADER']['BATCH_SIZE']
    dataset = FeatureMapDataset(cfg, is_train=is_train)                                             
    return dataset


if __name__=="__main__" :
    cfg_path = ""
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    data_loader = build_featuremap_dataloader(cfg)
    A = data_loader.__getitem__(0)
    import pdb;pdb.set_trace()