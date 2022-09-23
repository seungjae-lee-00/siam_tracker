#!/bin/sh
cfg=config/config.yaml
# model_path=
pretrained_path=/media/hdd/leetop/projects/siam_tracker/train_dir/SiamMOT_only_training_temp_2/model_final.pth
# pretrained_path= 
logging=train_dir/

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch train.py \
--config-file ${cfg} \
--train-dir ${logging} \
--pretrained ${pretrained_path}
# --local_rank 2
# --model-file ${model_path} \
