import torch
import torch.nn as nn

import torch.nn.functional as F
from torchvision.models import segmentation

from .unet_plus_plus import UnetPlusPlus


class FusionConvModel(nn.Module):
    def __init__(self):
        super(FusionConvModel, self).__init__()
        self.unetplusplus = UnetPlusPlus(num_classes=1)
        # self.classifier = self.deeplab.classifier

        deeplab = segmentation.deeplabv3_resnet50(aux_loss=True)
        deeplab.classifier[-1] = torch.nn.Conv2d(
            deeplab.classifier[-1].in_channels,
            1,
            kernel_size=deeplab.classifier[-1].kernel_size,
        )  # change number of outputs to 1
        self.deeplab = deeplab

        print(deeplab)
        print(self.unetplusplus)

        self.conv1 = nn.Conv2d(2, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 1, kernel_size=3, padding=1)

    def forward(self, x):
        deeplab_out = self.deeplab(x)["out"]
        unet_out = self.unetplusplus(x)

        deeplab_out = F.softmax(deeplab_out)
        unet_out = F.softmax(unet_out)

        print("deeplab_out -=====>", deeplab_out)
        print("unet_out -=====>", unet_out)

        out = self.merge_output(deeplab_out, unet_out)
        result = {"out": out}
        return result

    def merge_output(self, deeplab_out, unet_out):
        merged = torch.cat((deeplab_out, unet_out), dim=1)
        merged = self.conv1(merged)
        merged = self.conv2(merged)
        return merged
        # return (deeplab_out + unet_out) / 2


# def main():
#     model = FusionModel()
#     print(model.backbone["out"].out_channels)
#
#
# if __name__ == "__main__":
#     main()
