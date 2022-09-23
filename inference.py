import os
import argparse, logging, yaml, tqdm
import torch, torchvision, cv2
print(torch.cuda.device_count())
import model.SiamTracker
import utils.checkpointer
import utils.plot_box
from collections import OrderedDict
from PIL import Image
import pickle
from utils.bbox import BoxList
import utils.track_utils
import numpy as np
import cv2
import utils.plot_box


class Inference:
    def __init__(self, model_instance):
        self.model = model_instance.cuda()
        
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        utils.checkpointer.load_state_dict(self.model, checkpoint.pop("model"))
        self.model.eval()

    def _preprocess(self, pkl_path):        
        data = self.load_pkl(pkl_path)
        return torch.from_numpy(data["feature_map"]), torch.from_numpy(data["prev_template_feature"]), data["prev_detections"], data["prev_cls_labels"]

    def process(self, pkl_path, out_dir):
        in_featuremap, template_feature, detections, labels = self._preprocess(pkl_path)
        detections = self.convert_to_BoxList(detections, labels)
        sr = self.model.track_utils.update_boxes_in_pad_images([detections])
        sr = self.model.track_utils.extend_bbox(sr)
        # t_features = torch.from_numpy(template_feature).cuda()
        track_memory = (template_feature.cuda(), sr, [detections])
        
        # ---------- For Debug ---------- #
        img_name = os.path.basename(pkl_path)
        img_name = img_name.replace("pkl","jpg")
        img_path = "/media/hdd/leetop/data/waymo/testset_selected/"
        img_name = os.path.join(img_path, img_name)

        verify_data(img_name, "detected_input.jpg", detections)
        result = self.model(in_featuremap, proposals=None, targets=None, track_memory=track_memory)
        # result = self.model(in_featuremap, proposals=detections, targets=None, track_memory=track_memory)
        verify_data(img_name, "detected.jpg", result[0])
        return result

    @staticmethod
    def load_pkl(pkl_file):
        with open(pkl_file, 'rb') as f:
            u_data = pickle._Unpickler(f)
            u_data.encoding = "latin1"
            data = u_data.load()
            return data
    
    @staticmethod
    def convert_to_BoxList(objects, labels):
        # cls_labels = []
        # bbox_list = []
        # num_objects = 0

        # for cls_idx, box in enumerate(objects) :
        #     if cls_idx != 0:
        #         num_objects += len(box)
        
        # output = np.zeros((num_objects, 4), dtype=np.float32)
        # import pdb;pdb.set_trace()
        # idx = 0
        # for cls_idx, box in enumerate(objects):
        #     if cls_idx != 0:
        #         for xyxy in box:
        #             import pdb;pdb.set_trace() 
        #             output[idx][1:5] = xyxy[0:4]
        #             idx += 1
        #             cls_labels.append(cls_idx)

        boxes = torch.as_tensor(objects[:,1:5].tolist(), dtype=torch.float).reshape(-1,4).cuda()
        detections = BoxList(boxes, (1152, 768), mode="xyxy")            
        detections.add_field("labels", torch.as_tensor(labels, dtype=torch.int64))
        return detections


def verify_data(img_name, out_name, target):
    img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (1152, 768))
    for idx, bbox in enumerate(target.bbox.tolist()):
    # for idx, bbox in enumerate(target):
        track_id = target.get_field('labels')[idx]
        utils.plot_box.plot_one_box(bbox, img, label="track_id : {}".format(track_id), line_thickness=1)
    cv2.imwrite(out_name, img)


def main():
    # device = torch.device(f'cuda:{}')
    # torch.cuda.set_device(device)
    parser = argparse.ArgumentParser(description="PyTorch Video Object Detection Inference")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--dataset-dir", default="", help="path to test dataset", type=str)
    parser.add_argument("--output-dir", default="", help="path to output folder", type=str)
    parser.add_argument("--model-file", default=None, metavar="FILE", help="path to model file", type=str)
    parser.add_argument("--logging-dir", default="", help="directory for logging", type=str)
    parser.add_argument("--video-id", default="", type=str)

    args = parser.parse_args()
    
    with open(args.config_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    tracker_model = model.SiamTracker.SiamTracker(cfg)
    infer = Inference(tracker_model)

    fnames = [filename for filename in os.listdir(args.dataset_dir) if os.path.isfile(os.path.join(args.dataset_dir, filename))]
    file_list = [os.path.join(args.dataset_dir, fname) for fname in fnames if args.video_id in fname]
    file_list = sorted(file_list)
    infer.load_model(args.model_file)

    with torch.no_grad() :
        for idx, fname in tqdm.tqdm(enumerate(file_list)):
            if idx != 0:
                result = infer.process(fname, args.output_dir)


if __name__ == "__main__":
    main()