import os
import time
import torch
import numpy as np

from .epoch import run_epoch
import echonet


def run_train(
    model,
    optim,
    scheduler,
    batch_size,
    num_workers,
    num_epochs,
    device,
    kwargs,
    num_train_patients,
    data_dir,
    output,
    f,
):
    # Set up datasets and dataloaders
    dataset = {}
    dataset["train"] = echonet.datasets.Echo(root=data_dir, split="train", **kwargs)
    if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
        # Subsample patients (used for ablation experiment)
        indices = np.random.choice(
            len(dataset["train"]), num_train_patients, replace=False
        )
        dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)
    dataset["val"] = echonet.datasets.Echo(root=data_dir, split="val", **kwargs)

    # with open(os.path.join(output, "log.csv"), "a") as f:
    epoch_resume = 0
    bestLoss = float("inf")
    try:
        # Attempt to load checkpoint
        print("---------------------->", output)
        checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
        model.load_state_dict(checkpoint["state_dict"])
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
            ) = run_epoch(model, dataloader, phase == "train", optim, device)
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

        # Save checkpoint
        save = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "best_loss": bestLoss,
            "loss": loss,
            "opt_dict": optim.state_dict(),
            "scheduler_dict": scheduler.state_dict(),
        }
        torch.save(save, os.path.join(output, "checkpoint.pt"))
        if loss < bestLoss:
            torch.save(save, os.path.join(output, "best.pt"))
            bestLoss = loss
