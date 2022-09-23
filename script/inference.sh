#!/bin/sh
cfg=config/config.yaml
model_path=/media/hdd/leetop/projects/siam_tracker/train_dir/SiamMOT_only_training_temp_2/model_final.pth
dataset_path=/media/hdd/leetop/data/waymo/testset_add_prev
out_dir=outputs/
vid_id=1995626124544469658
# img_dir=/media/hdd/leetop/projects/ODwithTrack/vid_input

CUDA_VISIBLE_DEVICES=3 python inference.py \
--config-file ${cfg} \
--model-file ${model_path} \
--dataset-dir ${dataset_path} \
--output-dir ${out_dir} \
--video-id ${vid_id}
