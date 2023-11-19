import os
import torch
import time
from .epoch_kd import run_epoch_kd


# echonet.distill.train(output, model, dataloader, optim, device)
def run_train_kd(
    output,
    teacher,
    student,
    optim,
    dataset,
    batch_size,
    num_workers,
    num_epochs,
    # dataloader,
    lr_step_period,
    device,
    f,
):
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # with open(os.path.join(output, "log.distill.csv"), "a") as f:
    epoch_resume = 0
    bestLoss = float("inf")
    try:
        # Attempt to load checkpoint
        checkpoint = torch.load(os.path.join(output, "checkpoint.distill.pt"))
        student.load_state_dict(checkpoint["state_dict"])
        optim.load_state_dict(checkpoint["opt_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_dict"])
        epoch_resume = checkpoint["epoch"] + 1
        bestLoss = checkpoint["best_loss"]
        f.write("Resuming from epoch {}\n".format(epoch_resume))
    except FileNotFoundError:
        f.write("Starting run from scratch\n")

    for epoch in range(epoch_resume, num_epochs):
        print("Epoch #{}".format(epoch), flush=True)

        for phase in ["train", "val"]:
            start_time = time.time()
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)

            ds = dataset[phase]
            dataloader = torch.utils.data.DataLoader(
                ds,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                pin_memory=(device.type == "cuda"),
                drop_last=(phase == "train"),
            )
            (
                loss,
                large_inter,
                large_union,
                small_inter,
                small_union,
            ) = run_epoch_kd(teacher, student, dataloader, optim, device)

            overall_dice = (
                2
                * (large_inter.sum() + small_inter.sum())
                / (
                    large_union.sum()
                    + large_inter.sum()
                    + small_union.sum()
                    + small_inter.sum()
                )
            )
            large_dice = 2 * large_inter.sum() / (large_union.sum() + large_inter.sum())
            small_dice = 2 * small_inter.sum() / (small_union.sum() + small_inter.sum())
            f.write(
                "{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    epoch,
                    phase,
                    loss,
                    overall_dice,
                    large_dice,
                    small_dice,
                    time.time() - start_time,
                    large_inter.size,
                    sum(
                        torch.cuda.max_memory_allocated()
                        for i in range(torch.cuda.device_count())
                    ),
                    sum(
                        torch.cuda.max_memory_reserved()
                        for i in range(torch.cuda.device_count())
                    ),
                    batch_size,
                )
            )
            f.flush()
            scheduler.step()

        print(bestLoss)
