import argparse
import os, sys
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
print(torch.cuda.device_count())
import yaml
import logging
import model.SiamTracker
from trainer.tensorboard_writer import TensorboardWriter
from trainer.trainer import trainer
from utils.checkpointer import Checkpointer, load_state_dict
import utils.lr_scheduler
import data.build_dataset



parser = argparse.ArgumentParser(description="PyTorch SiamMOT tracker-only Training")
parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file", type=str)
parser.add_argument("--train-dir", default="", help="training folder where training artifacts are dumped", type=str)
parser.add_argument("--pretrained", default="", help="pretrained tracker model path", type=str)
parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

def setup_logger(name, save_dir, filename="trackert_only_log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def setup_env_and_logger(args, cfg):
    train_dir = os.path.join(args.train_dir, 'SiamMOT_only_training_FineTune')
    if train_dir:
        try:
            os.makedirs(train_dir)
        except OSError as e:
            import errno
            if e.errno != errno.EEXIST:
                raise

    logger = setup_logger("SiamMOT tracker", train_dir)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info(cfg)

    return train_dir, logger


def read_cfg(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        return cfg


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg['SOLVER']['BASE_LR']
        weight_decay = cfg['SOLVER']['WEIGHT_DECAY']
        if "bias" in key:
            lr = cfg['SOLVER']['BASE_LR'] * cfg['SOLVER']['BIAS_LR_FACTOR']
            weight_decay = cfg['SOLVER']['WEIGHT_DECAY_BIAS']
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg['SOLVER']['MOMENTUM'])
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return utils.lr_scheduler.WarmupMultiStepLR(
        optimizer,
        cfg['SOLVER']['STEPS'],
        cfg['SOLVER']['GAMMA'],
        warmup_factor=cfg['SOLVER']['WARMUP_FACTOR'],
        warmup_iters=cfg['SOLVER']['WARMUP_ITERS'],
        warmup_method=cfg['SOLVER']['WARMUP_METHOD']
    )

def train(cfg, train_dir, logger, pretrained_track=None):
    track_model = model.SiamTracker.SiamTracker(cfg)

    if pretrained_track!='':
        track_state_dict = torch.load(pretrained_track, map_location=torch.device("cpu"))
        new_model_dict = track_model.state_dict()
        pretrained_dict = {k: v for k, v in track_state_dict.items() if k in new_model_dict}
        new_model_dict.update(pretrained_dict)
        track_model.load_state_dict(new_model_dict)

    optimizer = make_optimizer(cfg, track_model)
    lr_scheduler = make_lr_scheduler(cfg, optimizer)
    train_dataset = data.build_dataset.build_featuremap_dataset(cfg)
    tensorboard_writer = TensorboardWriter(cfg, train_dir)
    checkpointer = Checkpointer(track_model, optimizer, lr_scheduler, train_dir, save_to_disk=True)

    torch.manual_seed(7)
    
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    track_model = torch.nn.parallel.DistributedDataParallel(track_model.cuda(), device_ids=[device_id])
    sampler_train = DistributedSampler(train_dataset)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, cfg['DATALOADER']['BATCH_SIZE'], drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                              batch_sampler=batch_sampler_train,
                                              num_workers=cfg['DATALOADER']['NUM_WORKERS'],
                                              collate_fn=data.build_dataset.sequence_featuremap_collate)
    
    # data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
    #                                collate_fn=utils.collate_fn, num_workers=args.num_workers)

    trainer(track_model, 
            train_loader,
            optimizer, 
            lr_scheduler, 
            checkpointer, 
            tensorboard_writer,
            logger
            )
    
    return track_model

def main():
    args = parser.parse_args()
    print(args)
    cfg = read_cfg(args.config_file)
    train_dir, logger = setup_env_and_logger(args, cfg)
    train(cfg, train_dir, logger, pretrained_track=args.pretrained)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()