"""Functions for training and running segmentation."""

import math
import os

import click
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import skimage.draw
import torch
import torchvision
import tqdm

from echonet.models.fusion_model import FusionModel
from echonet.models.resnet18_model import ResNet18Model

from .train_kd import run_train_kd

# from torchinfo import summary

import echonet


@click.command("segmentation")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option(
    "--model_name",
    type=click.Choice(
        sorted(
            name
            for name in torchvision.models.segmentation.__dict__
            if name.islower()
            and not name.startswith("__")
            and callable(torchvision.models.segmentation.__dict__[name])
        )
        + ["fusion"]
    ),
    default="deeplabv3_resnet50",
)
@click.option("--pretrained/--random", default=False)
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--run_test/--skip_test", default=False)
@click.option("--save_video/--skip_video", default=False)
@click.option("--num_epochs", type=int, default=50)
@click.option("--lr", type=float, default=1e-5)
@click.option("--weight_decay", type=float, default=0)
@click.option("--lr_step_period", type=int, default=None)
@click.option("--num_train_patients", type=int, default=None)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=20)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
@click.option("--run_distill/--skip_distill", default=False)
def run(
    data_dir=None,
    output=None,
    model_name="deeplabv3_resnet50",
    pretrained=False,
    weights=None,
    run_test=False,
    save_video=False,
    num_epochs=50,
    lr=1e-5,
    weight_decay=1e-5,
    lr_step_period=None,
    num_train_patients=None,
    num_workers=4,
    batch_size=20,
    device=None,
    seed=0,
    run_distill=False,
):
    """Trains/tests segmentation model.

    Args:
        data_dir (str, optional): Directory containing dataset. Defaults to
            `echonet.config.DATA_DIR`.
        output (str, optional): Directory to place outputs. Defaults to
            output/segmentation/<model_name>_<pretrained/random>/.
        model_name (str, optional): Name of segmentation model. One of ``deeplabv3_resnet50'',
            ``deeplabv3_resnet101'', ``fcn_resnet50'', or ``fcn_resnet101''
            (options are torchvision.models.segmentation.<model_name>)
            Defaults to ``deeplabv3_resnet50''.
        pretrained (bool, optional): Whether to use pretrained weights for model
            Defaults to False.
        weights (str, optional): Path to checkpoint containing weights to
            initialize model. Defaults to None.
        run_test (bool, optional): Whether or not to run on test.
            Defaults to False.
        save_video (bool, optional): Whether to save videos with segmentations.
            Defaults to False.
        num_epochs (int, optional): Number of epochs during training
            Defaults to 50.
        lr (float, optional): Learning rate for SGD
            Defaults to 1e-5.
        weight_decay (float, optional): Weight decay for SGD
            Defaults to 0.
        lr_step_period (int or None, optional): Period of learning rate decay
            (learning rate is decayed by a multiplicative factor of 0.1)
            Defaults to math.inf (never decay learning rate).
        num_train_patients (int or None, optional): Number of training patients
            for ablations. Defaults to all patients.
        num_workers (int, optional): Number of subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.
        device (str or None, optional): Name of device to run on. Options from
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
            Defaults to ``cuda'' if available, and ``cpu'' otherwise.
        batch_size (int, optional): Number of samples to load per batch
            Defaults to 20.
        seed (int, optional): Seed for random number generator. Defaults to 0.
    """

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set default output directory
    if output is None:
        output = os.path.join(
            "output",
            "segmentation",
            "{}_{}".format(model_name, "pretrained" if pretrained else "random"),
        )
    os.makedirs(output, exist_ok=True)

    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = None
    if model_name == "fusion":
        model = FusionModel()
    else:
        model = torchvision.models.segmentation.__dict__[model_name](
            weights="DEFAULT" if pretrained else None,
            aux_loss=True if pretrained else False,
        )

    model.classifier[-1] = torch.nn.Conv2d(
        model.classifier[-1].in_channels,
        1,
        kernel_size=model.classifier[-1].kernel_size,
    )  # change number of outputs to 1
    print("model classifier last layer changed")

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    if weights is not None:
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint["state_dict"])

    # Set up optimizer
    optim = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
    )
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # Compute mean and std
    mean, std = echonet.utils.get_mean_and_std(
        echonet.datasets.Echo(root=data_dir, split="train")
    )
    tasks = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
    kwargs = {"target_type": tasks, "mean": mean, "std": std}

    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:
        echonet.utils.run_train(
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
        )

        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint["state_dict"])
            f.write(
                "Best validation loss {} from epoch {}\n".format(
                    checkpoint["loss"], checkpoint["epoch"]
                )
            )

        if run_test:
            # Run on validation and test
            echonet.utils.run_test(
                model, batch_size, num_workers, device, kwargs, data_dir, output, f
            )

    if run_distill:
        # student = echonet.models.deeplabv3_restnet50(
        #     num_classes=7, aux_loss=True if pretrained else False
        # )
        # student = echonet.models.restnet50()
        student = ResNet18Model()

        if device.type == "cuda":
            student = torch.nn.DataParallel(student)
        student.to(device)

        # Set up optimizer
        optim = torch.optim.SGD(
            student.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
        if lr_step_period is None:
            lr_step_period = math.inf
        scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

        with open(os.path.join(output, "log.distill.csv"), "a") as f:
            print("KD run train")
            run_train_kd(
                model,
                student,
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
            )

            # Load best weights
            if num_epochs != 0:
                checkpoint = torch.load(os.path.join(output, "best.distill.pt"))
                student.load_state_dict(checkpoint["state_dict"])
                f.write(
                    "Distill Best validation loss {} from epoch {}\n".format(
                        checkpoint["loss"], checkpoint["epoch"]
                    )
                )

            if run_test:
                # Run on validation and test
                print("KD run test")
                echonet.utils.run_test(
                    student,
                    batch_size,
                    num_workers,
                    device,
                    kwargs,
                    data_dir,
                    output,
                    f,
                )

    # Saving videos with segmentations
    dataset = echonet.datasets.Echo(
        root=data_dir,
        split="test",
        target_type=[
            "Filename",
            "LargeIndex",
            "SmallIndex",
        ],  # Need filename for saving, and human-selected frames to annotate
        mean=mean,
        std=std,  # Normalization
        length=None,
        max_length=None,
        period=1,  # Take all frames
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,
        collate_fn=_video_collate_fn,
    )

    # Save videos with segmentation
    if save_video and not all(
        os.path.isfile(os.path.join(output, "videos", f))
        for f in dataloader.dataset.fnames
    ):
        # Only run if missing videos

        model.eval()

        os.makedirs(os.path.join(output, "videos"), exist_ok=True)
        os.makedirs(os.path.join(output, "size"), exist_ok=True)
        echonet.utils.latexify()

        with torch.no_grad():
            with open(os.path.join(output, "size.csv"), "w") as g:
                g.write("Filename,Frame,Size,HumanLarge,HumanSmall,ComputerSmall\n")
                for x, (filenames, large_index, small_index), length in tqdm.tqdm(
                    dataloader
                ):
                    # Run segmentation model on blocks of frames one-by-one
                    # The whole concatenated video may be too long to run together
                    y = np.concatenate(
                        [
                            model(x[i : (i + batch_size), :, :, :].to(device))["out"]
                            .detach()
                            .cpu()
                            .numpy()
                            for i in range(0, x.shape[0], batch_size)
                        ]
                    )

                    start = 0
                    x = x.numpy()
                    for i, (filename, offset) in enumerate(zip(filenames, length)):
                        # Extract one video and segmentation predictions
                        video = x[start : (start + offset), ...]
                        logit = y[start : (start + offset), 0, :, :]

                        # Un-normalize video
                        video *= std.reshape(1, 3, 1, 1)
                        video += mean.reshape(1, 3, 1, 1)

                        # Get frames, channels, height, and width
                        f, c, h, w = video.shape  # pylint: disable=W0612
                        assert c == 3

                        # Put two copies of the video side by side
                        video = np.concatenate((video, video), 3)

                        # If a pixel is in the segmentation, saturate blue channel
                        # Leave alone otherwise
                        video[:, 0, :, w:] = np.maximum(
                            255.0 * (logit > 0), video[:, 0, :, w:]
                        )  # pylint: disable=E1111

                        # Add blank canvas under pair of videos
                        video = np.concatenate((video, np.zeros_like(video)), 2)

                        # Compute size of segmentation per frame
                        size = (logit > 0).sum((1, 2))

                        # Identify systole frames with peak detection
                        trim_min = sorted(size)[round(len(size) ** 0.05)]
                        trim_max = sorted(size)[round(len(size) ** 0.95)]
                        trim_range = trim_max - trim_min
                        systole = set(
                            scipy.signal.find_peaks(
                                -size, distance=20, prominence=(0.50 * trim_range)
                            )[0]
                        )

                        # Write sizes and frames to file
                        for frame, s in enumerate(size):
                            g.write(
                                "{},{},{},{},{},{}\n".format(
                                    filename,
                                    frame,
                                    s,
                                    1 if frame == large_index[i] else 0,
                                    1 if frame == small_index[i] else 0,
                                    1 if frame in systole else 0,
                                )
                            )

                        # Plot sizes
                        fig = plt.figure(figsize=(size.shape[0] / 50 * 1.5, 3))
                        plt.scatter(np.arange(size.shape[0]) / 50, size, s=1)
                        ylim = plt.ylim()
                        for s in systole:
                            plt.plot(np.array([s, s]) / 50, ylim, linewidth=1)
                        plt.ylim(ylim)
                        plt.title(os.path.splitext(filename)[0])
                        plt.xlabel("Seconds")
                        plt.ylabel("Size (pixels)")
                        plt.tight_layout()
                        plt.savefig(
                            os.path.join(
                                output, "size", os.path.splitext(filename)[0] + ".pdf"
                            )
                        )
                        plt.close(fig)

                        # Normalize size to [0, 1]
                        size -= size.min()
                        size = size / size.max()
                        size = 1 - size

                        # Iterate the frames in this video
                        for f, s in enumerate(size):
                            # On all frames, mark a pixel for the size of the frame
                            video[
                                :,
                                :,
                                int(round(115 + 100 * s)),
                                int(round(f / len(size) * 200 + 10)),
                            ] = 255.0

                            if f in systole:
                                # If frame is computer-selected systole, mark with a line
                                video[
                                    :, :, 115:224, int(round(f / len(size) * 200 + 10))
                                ] = 255.0

                            def dash(start, stop, on=10, off=10):
                                buf = []
                                x = start
                                while x < stop:
                                    buf.extend(range(x, x + on))
                                    x += on
                                    x += off
                                buf = np.array(buf)
                                buf = buf[buf < stop]
                                return buf

                            d = dash(115, 224)

                            if f == large_index[i]:
                                # If frame is human-selected diastole, mark with green dashed line on all frames
                                video[
                                    :, :, d, int(round(f / len(size) * 200 + 10))
                                ] = np.array([0, 225, 0]).reshape((1, 3, 1))
                            if f == small_index[i]:
                                # If frame is human-selected systole, mark with red dashed line on all frames
                                video[
                                    :, :, d, int(round(f / len(size) * 200 + 10))
                                ] = np.array([0, 0, 225]).reshape((1, 3, 1))

                            # Get pixels for a circle centered on the pixel
                            r, c = skimage.draw.disk(
                                (
                                    int(round(115 + 100 * s)),
                                    int(round(f / len(size) * 200 + 10)),
                                ),
                                4.1,
                            )

                            # On the frame that's being shown, put a circle over the pixel
                            video[f, :, r, c] = 255.0

                        # Rearrange dimensions and save
                        video = video.transpose(1, 0, 2, 3)
                        video = video.astype(np.uint8)
                        echonet.utils.savevideo(
                            os.path.join(output, "videos", filename), video, 50
                        )

                        # Move to next video
                        start += offset


def _video_collate_fn(x):
    """Collate function for Pytorch dataloader to merge multiple videos.

    This function should be used in a dataloader for a dataset that returns
    a video as the first element, along with some (non-zero) tuple of
    targets. Then, the input x is a list of tuples:
      - x[i][0] is the i-th video in the batch
      - x[i][1] are the targets for the i-th video

    This function returns a 3-tuple:
      - The first element is the videos concatenated along the frames
        dimension. This is done so that videos of different lengths can be
        processed together (tensors cannot be "jagged", so we cannot have
        a dimension for video, and another for frames).
      - The second element is contains the targets with no modification.
      - The third element is a list of the lengths of the videos in frames.
    """
    video, target = zip(*x)  # Extract the videos and targets

    # ``video'' is a tuple of length ``batch_size''
    #   Each element has shape (channels=3, frames, height, width)
    #   height and width are expected to be the same across videos, but
    #   frames can be different.

    # ``target'' is also a tuple of length ``batch_size''
    # Each element is a tuple of the targets for the item.

    i = list(map(lambda t: t.shape[1], video))  # Extract lengths of videos in frames

    # This contatenates the videos along the the frames dimension (basically
    # playing the videos one after another). The frames dimension is then
    # moved to be first.
    # Resulting shape is (total frames, channels=3, height, width)
    video = torch.as_tensor(np.swapaxes(np.concatenate(video, 1), 0, 1))

    # Swap dimensions (approximately a transpose)
    # Before: target[i][j] is the j-th target of element i
    # After:  target[i][j] is the i-th target of element j
    target = zip(*target)

    return video, target, i
