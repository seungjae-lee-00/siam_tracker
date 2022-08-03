import cv2
import random
import numpy as np
import utils.plot_box

colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
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


def display_boxes(img, boxlist):

    for boxes in boxlist:
        for box in boxes.bbox:
            # import pdb;pdb.set_trace()
            xyxy = box.detach().cpu().numpy()
            xyxy[0] = xyxy[0]*img.shape[1]/1280
            xyxy[1] = xyxy[1]*img.shape[0]/704
            xyxy[2] = xyxy[2]*img.shape[1]/1280
            xyxy[3] = xyxy[3]*img.shape[0]/704
            plot_one_box(xyxy, img)
    return img

def boxes_on_image(images, boxes, out_name):
    mean = np.asarray([ 0.485, 0.456, 0.406 ])
    std = np.asarray([ 0.229, 0.224, 0.225 ])

    for idx, image in enumerate(images):
        img = image.detach().cpu().numpy()
        img = np.transpose(img, (1,2,0))
        img = np.clip(255.0*(img*std+mean), 0, 255)
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for cnt, box in enumerate(boxes[idx].bbox):
            if boxes[idx].has_field("scores"):
                if boxes[idx].get_field("scores")[cnt] > 0.5:
                    if boxes[idx].get_field("ids")[cnt] > -1:
                        label = "obj_id : {}".format(boxes[idx].get_field("ids")[cnt])
                        utils.plot_box.plot_one_box(box, img, color=colors[int(boxes[idx].get_field("ids")[cnt])%100], label=label, line_thickness=1)
            else :                
                # if boxes[idx].get_field("ids")[cnt] > -1:
                    # label = "obj_id : {}".format(boxes[idx].get_field("ids")[cnt])
                utils.plot_box.plot_one_box(box, img, line_thickness=1)
            # utils.plot_box.plot_one_box(box, img, color=colors[int(boxes[idx].get_field("ids")[cnt])], line_thickness=1)

        cv2.imwrite('{0}.jpg'.format(out_name), img)

def boxes_on_image_with_scores(images, boxes, out_name):
    mean = np.asarray([ 0.485, 0.456, 0.406 ])
    std = np.asarray([ 0.229, 0.224, 0.225 ])

    for idx, image in enumerate(images):
        img = image.detach().cpu().numpy()
        img = np.transpose(img, (1,2,0))
        img = np.clip(255.0*(img*std+mean), 0, 255)
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for cnt, box in enumerate(boxes[idx].bbox):
            label = boxes[idx].get_field("ids")[cnt]
            score = boxes[idx].get_field("scores")[cnt]
            if score > 0.5:
                info_str = "ID:{0}, score:{1:3f}".format(label, score)
                utils.plot_box.plot_one_box(box, img, color=colors[int(boxes[idx].get_field("ids")[cnt])%100], label=info_str, line_thickness=1)

        cv2.imwrite('{0}.jpg'.format(out_name), img)
