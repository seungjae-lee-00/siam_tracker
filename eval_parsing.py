import os
import pickle
from model.track_head.sr_pool import SRPooler
import numpy as np
import torch, torchvision

def load_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u_data = pickle._Unpickler(f)
        u_data.encoding = "latin1"
        data = u_data.load()
        return data    


def pooler(resolution, scale, sampling_ratio):
    pooler_z = torchvision.ops.RoIAlign(
                    (resolution, resolution), spatial_scale=scale, sampling_ratio=sampling_ratio
                )        
    return pooler_z


def convert_to_rois(boxes):
    num_objects = 0
    
    for cls_idx, box in enumerate(boxes) :
        if cls_idx != 0:
            # num_objects += len(box)
            num_objects += len(box[box[:,4]>0.8])
    
    output = np.zeros((num_objects, 5), dtype=np.float32)

    idx = 0
    cls_labels = []

    for cls_idx, box in enumerate(boxes):
        if cls_idx != 0:
            for xyxy in box:
                score = xyxy[4]
                if score>0.8:
                    cls_labels.append(cls_idx)
                    output[idx, 1] = xyxy[0] * 1152/1920
                    output[idx, 2] = xyxy[1] * 768/1280
                    output[idx, 3] = xyxy[2] * 1152/1920
                    output[idx, 4] = xyxy[3] * 768/1280
    
                    idx += 1

    return output, cls_labels



def main():
    pkl_root = "/media/hdd/leetop/data/waymo/testset_extracted"
    out_dir = "/media/hdd/leetop/data/waymo/testset_add_prev"

    file_list = [f for f in os.listdir(pkl_root) if os.path.isfile(os.path.join(pkl_root, f))]
    file_list = sorted(file_list)
    template_pooler = pooler(15, 0.125, 2)
    prev_template_features = None
    prev_detections = None
    cls_labels = []

    for fname in file_list:
        data = load_pickle(os.path.join(pkl_root, fname))
        data["prev_template_feature"] = prev_template_features
        data["prev_detections"] = prev_detections
        data["prev_cls_labels"] = cls_labels
        with open(os.path.join(out_dir, fname), 'wb') as f:
             pickle.dump(data, f)
        
        rois, cls_labels = convert_to_rois(data["objects"])
        prev_detections = rois
        feature_map = torch.from_numpy(data["feature_map"])
        rois = torch.from_numpy(rois)
        template_feature = template_pooler(feature_map, rois)
        prev_template_features = template_feature.numpy()

if __name__=="__main__" :
    main()
