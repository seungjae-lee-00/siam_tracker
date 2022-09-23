import datetime
import time
# from typing_extensions import dataclass_transform

from utils.metrics import MetricLogger
import torch

torch.autograd.set_detect_anomaly(True)

def trainer(model_instance, data_loader, optimizer, scheduler, checkpointer, tensorboard_writer, logger):
    torch.cuda.empty_cache()

    logger.info("------- start training -------")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    model_instance.cuda().train()

    logger.info('------- model info. -------')
    logger.info(model_instance)

    start_training_time = time.time()
    end = time.time()

    for epoch in range(15):
        for iteration, (features, proposals, targets) in enumerate(data_loader):
            data_time = time.time() - end
            result, loss_dict = model_instance(features, proposals, targets)

            optimizer.zero_grad()
            # losses = sum(loss for loss in loss_dict.values())
            losses = sum(loss_dict.values())
            losses.backward()
            meters.update(loss=losses, **loss_dict)
            optimizer.step()
            scheduler.step()
            # tensorboard_writer(iteration+(epoch*max_iter), losses, loss_dict, targets)

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 10 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=max_iter*(epoch)+iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                    )
                )
            if (iteration-1) % 500 == 0 :
                checkpointer.save("model_{:07d}".format(max_iter*(epoch)+iteration))


        logger.info(
            meters.delimiter.join(
                [
                    "eta: {eta}",
                    "epoch: {epoch}",
                    "{meters}",
                    "lr: {lr:.6f}",
                ]
            ).format(
                eta=eta_string,
                epoch=epoch+1,
                meters=str(meters),
                lr=optimizer.param_groups[0]["lr"],
            )
        )

    checkpointer.save("model_final".format(max_iter*(epoch+1)+iteration))

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )