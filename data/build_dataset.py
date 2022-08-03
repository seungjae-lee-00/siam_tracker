import os, json

import itertools
import torch
import torch.nn.functional as F
import torchvision

from random import shuffle
from PIL import Image, ImageDraw
import pandas as pd
import xml.etree.ElementTree as elemTree

from utils.bbox import BoxList
from model.transforms import build_transforms

import utils.plot_box

# ---- for 4 classes ----
cls_list = ['pedestrian', 'rider_bicycle', 'rider_bike', 'bicycle', 'bike', 
            '3-wheels_rider', '3-wheels', 'sedan', 'van', 'pickup_truck', 
            'truck', 'mixer_truck', 'excavator', 'forklift', 'ladder_truck', 
            'truck_etc', 'vehicle_etc', 'vehicle_special', 'box_truck', 'trailer', 
            'bus', 'ehicle_special', 'sitting_person', 'wheel_chair', 'ignored', 
            'false_positive', 'animal', 'bird', 'animal_ignored']

cls_index = [1, 3, 3, -1, -1, 
             3, -1, 2, 2, 2, 
             2, 2, 2, 2, 2, 
             2, 2, 2, 2, 2, 
             2, -1, -1, -1, -1,
             -1, -1, -1, -1, -1]

# --- for 8 classes ----
# cls_list = ['pedestrian', 'rider_bicycle', 'rider_bicycle_2', 'rider_bike', 'rider_bike_2', 
#             'rider_bicycle_human_body', 'rider_bike_human_body', 'bicycle',    'bike', '3-wheels_rider', 
#             '3-wheels', 'sedan', 'van', 'pickup_truck', 'truck', 
#             'mixer_truck', 'excavator', 'forklift', 'ladder_truck', 'truck_etc', 
#             'vehicle_etc', 'vehicle_special', 'box_truck', 'trailer', 'bus', 
#             'ehicle_special', 'sitting_person', 'wheel_chair', 'ignored', 'false_positive', 
#             'animal', 'bird', 'animal_ignored', 'ts_circle', 'ts_circle_speed', 
#             'ts_triangle', 'ts_inverted_triangle', 'ts_rectangle', 'ts_rectangle_speed', 'ts_diamonds', 
#             'ts_supplementary', 'tl_car',  'tl_ped',  'tl_special', 'tl_light_only', 
#             'ts_ignored', 'tl_ignored', 'tstl_ignore', 'ts_sup_ignored', 'ts_sup_letter', 
#             'ts_sup_drawing', 'ts_sup_arrow', 'ts_sup_zone', 'ts_main_zone', 'ts_rectangle_arrow']
# cls_index = [1, 2, 2, 3, 3,
#             -1, -1, -1, -1, 3, 
#             -1, 4, 4, 4, 5, 
#             5, 5, 5, 5, 5, 
#             5, 5, 5, 5, 6, 
#             5, 1, 1, -1, -1, 
#             -1, -1, -1,  7, 7, 
#             7, 7, 7,  7, 7, 
#             -1, -1, -1, -1, -1, 
#             -1, -1, -1, -1, 7, 
#             7, 7, 7, -1, -1]

class SingleImageDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transforms=None, is_train=True):
        csv_file = cfg['DATASETS']['TRAIN'] if is_train else cfg['DATASETS']['TEST']
        self.data_root = cfg['DATASETS']['ROOT_DIR']
        self.df = pd.read_csv(os.path.join(self.data_root, csv_file))
        self.transforms = transforms
        self.device = torch.device(cfg['MODEL']['DEVICE'])
                
    def __getitem__(self, idx):
        fname = self.df.img[idx]
        ann_name = self.df.ann[idx]

        img = Image.open(os.path.join(self.data_root, 'JPEGImages/', fname)).convert("RGB")
        target = self.get_ann_info(img, ann_name)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return [img, target]

    def __len__(self):
        return len(self.df)

    def get_ann_info(self, img, ann_name):
        ann_path = os.path.join(self.data_root,'Annotations/', ann_name)
        tree = elemTree.parse(ann_path)
        objects = tree.findall('object')
        boxes = []
        cls_ids = []

        for obj in objects:
            cls_name = obj.find('name').text
            
            if cls_name in cls_list:
                cls_idx_val = cls_list.index(cls_name)
                if cls_index[cls_idx_val] != -1:
                    cls_id = torch.tensor(cls_index[cls_idx_val], dtype=torch.int64)
                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymin = float(bbox.find('ymin').text)
                    ymax = float(bbox.find('ymax').text)
                    boxes.append([xmin, ymin, xmax, ymax])
                    cls_ids.append(cls_id)
        
        boxes = torch.as_tensor(boxes, device=self.device).reshape(-1,4)
        target = BoxList(boxes, img.size, mode="xyxy")
        target.add_field("labels", torch.as_tensor(cls_ids, dtype=torch.int64))

        return target

def draw_boxes(image, point1, point2, color=(0,0,255)):
    draw = ImageDraw.Draw(image)
    draw.rectangle((point1, point2), outline=color, width=2)
    return image


class SequenceImageDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transforms=None, is_train=True):
        csv_file = cfg['DATASETS']['TRAIN'] if is_train else cfg['DATASETS']['TEST']
        self.data_root = cfg['DATASETS']['ROOT_DIR']
        self.df = pd.read_csv(os.path.join(self.data_root, csv_file))
        self.transforms = transforms
        self.device = torch.device(cfg['MODEL']['DEVICE'])

        # ---- for 8-classes ----
        # self.class_index = {'pedestrian' : 1,
        #                     'bicycle' : 2,
        #                     'motorbike' : 3,
        #                     'car' : 4,
        #                     'truck' : 5,
        #                     'bus' : 6,
        #                     # 'ts' : 7
        #                     # 'tl' : 8
        #                     }

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


def singleimage_collate(batch):
    transposed_batch = list(zip(*batch))
    image_batch = transposed_batch[0]
    targets = transposed_batch[1]
    
    return image_batch, targets

def sequence_image_collate(batch):
    images = []
    targets = []

    for data in batch:
        images.append(data[0][0])
        images.append(data[0][1])
        targets.append(data[1][0])
        targets.append(data[1][1])
        
    return images, targets

def build_dataloader(cfg, is_train=True):
    batch_size = cfg['DATALOADER']['BATCH_SIZE']
    transforms = build_transforms(cfg)
    dataset = SingleImageDataset(cfg, transforms=transforms, is_train=is_train)
    data_loader = torch.utils.data.DataLoader(dataset, 
                                              num_workers=cfg['DATALOADER']['NUM_WORKERS'],
                                              batch_size=batch_size,
                                              shuffle=True,
                                              collate_fn=singleimage_collate)
    return data_loader

def build_track_dataloader(cfg, is_train=True):
    batch_size = cfg['DATALOADER']['BATCH_SIZE']
    transforms = build_transforms(cfg, is_train=False)
    dataset = SequenceImageDataset(cfg, transforms=transforms, is_train=is_train)
    data_loader = torch.utils.data.DataLoader(dataset, 
                                              num_workers=cfg['DATALOADER']['NUM_WORKERS'],
                                              batch_size=batch_size,
                                              shuffle=True,
                                              collate_fn=sequence_image_collate)
                                              
    return data_loader
