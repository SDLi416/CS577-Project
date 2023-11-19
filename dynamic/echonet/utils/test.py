import os
import torch
import echonet
import numpy as np


def run_test(model, device, data_dir, output, num_workers, batch_size, kwargs, f):
    print("------------------------------------> run test")
    for split in ["val", "test"]:
        dataset = echonet.datasets.Echo(root=data_dir, split=split, **kwargs)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=(device.type == "cuda"),
        )
        (
            loss,
            large_inter,
            large_union,
            small_inter,
            small_union,
        ) = echonet.utils.run_epoch(model, dataloader, False, None, device)
        print(loss, large_inter, large_union, small_inter, small_union)

        overall_dice = (
            2
            * (large_inter + small_inter)
            / (large_union + large_inter + small_union + small_inter)
        )
        large_dice = 2 * large_inter / (large_union + large_inter)
        small_dice = 2 * small_inter / (small_union + small_inter)
        with open(os.path.join(output, "{}_dice.csv".format(split)), "w") as g:
            g.write("Filename, Overall, Large, Small\n")
            for filename, overall, large, small in zip(
                dataset.fnames, overall_dice, large_dice, small_dice
            ):
                g.write("{},{},{},{}\n".format(filename, overall, large, small))

        f.write(
            "{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format(
                split,
                *echonet.utils.bootstrap(
                    np.concatenate((large_inter, small_inter)),
                    np.concatenate((large_union, small_union)),
                    echonet.utils.dice_similarity_coefficient,
                ),
            )
        )
        f.write(
            "{} dice (large):   {:.4f} ({:.4f} - {:.4f})\n".format(
                split,
                *echonet.utils.bootstrap(
                    large_inter,
                    large_union,
                    echonet.utils.dice_similarity_coefficient,
                ),
            )
        )
        f.write(
            "{} dice (small):   {:.4f} ({:.4f} - {:.4f})\n".format(
                split,
                *echonet.utils.bootstrap(
                    small_inter,
                    small_union,
                    echonet.utils.dice_similarity_coefficient,
                ),
            )
        )
        f.flush()
