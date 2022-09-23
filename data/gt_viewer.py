import random
import cv2
import os
import json

def plot_one_box(x, img, color=None, label=None, line_thickness=2):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        
        
def gt_view(df):
    ann_root = "/media/hdd/leetop/data/waymo/M575_matching_final"
    img_root = "/media/hdd/leetop/data/waymo/JPEGImage"
    img1_list = df.img_1.to_list()
    img2_list = df.img_2.to_list()

    img_list = img1_list+img2_list

    for im in img_list:
        img_name = os.path.join(img_root, im)
        ann_name = os.path.join(ann_root, im.replace(".jpg", ".json"))
        
        with open(ann_name, 'r') as json_file:
            ann_data = json.load(json_file)
            
        img = cv2.imread(img_name)

        for obj in ann_data["objects"]:
            xyxy = obj['xyxy']
            cls_name = obj['class']
            plot_one_box(xyxy, img, label=cls_name, line_thickness=1)

        cv2.imwrite('test_ann.jpg', img)
        
def main():
    filename = "/media/hdd/leetop/data/waymo/JPEGImage/14734824171146590110_880_000_900_000_with_camera_labels_00000014_FRONT.jpg"
    # ann_name = os.path.basename(filename).replace('jpg','json')
    # ann_name = os.path.join("/media/hdd/leetop/data/waymo/M575_matched/",ann_name)
    
    # with open(ann_name,'r') as json_file:
    #     json_data = json.load(json_file)

    # img = cv2.imread(filename)
    
    # for obj in json_data['objects']:
    #     xyxy = obj['xyxy']
    #     cls_name = obj['class']
    #     plot_one_box(xyxy, img, label=cls_name)
    
    # cv2.imwrite('test_ann.jpg', img)

    ann_name = os.path.basename(filename).replace('jpg','pkl')
    ann_name = os.path.join("/media/hdd/leetop/data/waymo/M578_detected",ann_name)
    
    import pickle
    with open(ann_name,'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()

    # import pdb;pdb.set_trace()
    img = cv2.imread(filename)
    import pdb;pdb.set_trace()
    for obj in data["proposals"]:
        xyxy = obj[1:5]*5/3
        plot_one_box(xyxy, img)
    
    cv2.imwrite('test_ann.jpg', img)


if __name__ == "__main__":
    main()