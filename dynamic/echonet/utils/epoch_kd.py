import torch

import math
import numpy as np
import tqdm

# from .losses import DistillationMSELoss
from .losses import DistillationKLDivLoss


def run_epoch_kd(
    teacher,
    student,
    dataloader,
    train,
    optim,
    device,
    # T=2,
    # soft_target_loss_weight=0.25,
    # ce_loss_weight=0.75,
):
    total = 0.0
    n = 0

    pos = 0
    neg = 0
    pos_pix = 0
    neg_pix = 0

    # model.train(train)

    large_inter = 0
    large_union = 0
    small_inter = 0
    small_union = 0
    large_inter_list = []
    large_union_list = []
    small_inter_list = []
    small_union_list = []

    kd_loss_func = DistillationKLDivLoss()

    teacher.eval()
    if train:
        student.train()
    else:
        student.eval()

    print("run_epoch_kd ------------------>")

    # with torch.set_grad_enabled(train):
    with tqdm.tqdm(total=len(dataloader)) as pbar:
        for _, (large_frame, small_frame, large_trace, small_trace) in dataloader:
            # Count number of pixels in/out of human segmentation
            pos += (large_trace == 1).sum().item()
            pos += (small_trace == 1).sum().item()
            neg += (large_trace == 0).sum().item()
            neg += (small_trace == 0).sum().item()

            # Count number of pixels in/out of computer segmentation
            pos_pix += (large_trace == 1).sum(0).to("cpu").detach().numpy()
            pos_pix += (small_trace == 1).sum(0).to("cpu").detach().numpy()
            neg_pix += (large_trace == 0).sum(0).to("cpu").detach().numpy()
            neg_pix += (small_trace == 0).sum(0).to("cpu").detach().numpy()

            # Run prediction for diastolic frames and compute loss
            large_frame = large_frame.to(device)
            large_trace = large_trace.to(device)
            y_large = student(large_frame)["out"]
            # print("xxxx-xx-x-", y_large.shape)
            # print(y_large)

            # Compute pixel intersection and union between human and computer segmentations
            large_inter += np.logical_and(
                y_large[:, 0, :, :].detach().cpu().numpy() > 0.0,
                large_trace[:, :, :].detach().cpu().numpy() > 0.0,
            ).sum()
            large_union += np.logical_or(
                y_large[:, 0, :, :].detach().cpu().numpy() > 0.0,
                large_trace[:, :, :].detach().cpu().numpy() > 0.0,
            ).sum()
            large_inter_list.extend(
                np.logical_and(
                    y_large[:, 0, :, :].detach().cpu().numpy() > 0.0,
                    large_trace[:, :, :].detach().cpu().numpy() > 0.0,
                ).sum((1, 2))
            )
            large_union_list.extend(
                np.logical_or(
                    y_large[:, 0, :, :].detach().cpu().numpy() > 0.0,
                    large_trace[:, :, :].detach().cpu().numpy() > 0.0,
                ).sum((1, 2))
            )

            # Run prediction for systolic frames and compute loss
            small_frame = small_frame.to(device)
            small_trace = small_trace.to(device)
            y_small = student(small_frame)["out"]

            # Compute pixel intersection and union between human and computer segmentations
            small_inter += np.logical_and(
                y_small[:, 0, :, :].detach().cpu().numpy() > 0.0,
                small_trace[:, :, :].detach().cpu().numpy() > 0.0,
            ).sum()
            small_union += np.logical_or(
                y_small[:, 0, :, :].detach().cpu().numpy() > 0.0,
                small_trace[:, :, :].detach().cpu().numpy() > 0.0,
            ).sum()
            small_inter_list.extend(
                np.logical_and(
                    y_small[:, 0, :, :].detach().cpu().numpy() > 0.0,
                    small_trace[:, :, :].detach().cpu().numpy() > 0.0,
                ).sum((1, 2))
            )
            small_union_list.extend(
                np.logical_or(
                    y_small[:, 0, :, :].detach().cpu().numpy() > 0.0,
                    small_trace[:, :, :].detach().cpu().numpy() > 0.0,
                ).sum((1, 2))
            )

            with torch.no_grad():
                teacher_y_large = teacher(large_frame)["out"]
                teacher_y_small = teacher(small_frame)["out"]

            kd_loss_large = kd_loss_func(
                y_large[:, 0, :, :], teacher_y_large[:, 0, :, :], large_trace
            )
            kd_loss_small = kd_loss_func(
                y_small[:, 0, :, :], teacher_y_small[:, 0, :, :], small_trace
            )
            loss_kd = (kd_loss_large + kd_loss_small) / 2
            # loss = (loss_large + loss_small) / 2

            # Take gradient step if training
            if train:
                optim.zero_grad()
                # loss.backward()
                loss_kd.backward()
                optim.step()

            # Accumulate losses and compute baselines
            total += loss_kd.item()
            n += large_trace.size(0)
            p = pos / (pos + neg)
            p_pix = (pos_pix + 1) / (pos_pix + neg_pix + 2)

            # Show info on process bar
            pbar.set_postfix_str(
                "{:.4f} ({:.4f}) / {:.4f} {:.4f}, {:.4f}, {:.4f}".format(
                    total / n / 112 / 112,
                    loss_kd.item() / large_trace.size(0) / 112 / 112,
                    -p * math.log(p) - (1 - p) * math.log(1 - p),
                    (-p_pix * np.log(p_pix) - (1 - p_pix) * np.log(1 - p_pix)).mean(),
                    2 * large_inter / (large_union + large_inter),
                    2 * small_inter / (small_union + small_inter),
                )
            )
            pbar.update()
            # break

    large_inter_list = np.array(large_inter_list)
    large_union_list = np.array(large_union_list)
    small_inter_list = np.array(small_inter_list)
    small_union_list = np.array(small_union_list)

    return (
        total / n / 112 / 112,
        large_inter_list,
        large_union_list,
        small_inter_list,
        small_union_list,
    )
