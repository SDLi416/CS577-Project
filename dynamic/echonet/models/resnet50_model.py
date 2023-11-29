import torch.nn as nn

# from torchvision.models import segmentation
import torchvision


# def restnet50() -> torchvision.models.ResNet:
#     return torchvision.models.resnet50(replace_stride_with_dilation=[False, True, True])


class ResNet50Model(nn.Module):
    def __init__(self):
        super(ResNet50Model, self).__init__()
        self.resnet50 = torchvision.models.resnet50(
            replace_stride_with_dilation=[False, True, True]
        )
        self.line = nn.Linear(1000, 112 * 112)

    def forward(self, x):
        # deeplab_out = self.deeplab(x)["out"]
        # unet_out = self.unetplusplus(x)
        # out = self.merge_output(deeplab_out, unet_out)
        out = self.resnet50(x)
        out = self.line(out)
        out = out.reshape(out.shape[0], 1, 112, 112)
        result = {"out": out}
        return result
