import torch
from typing import Any, Optional
import torchvision

from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation import deeplabv3
from torchvision.models.segmentation.fcn import FCNHead


def deeplabv3_restnet50(
    num_classes: Optional[int] = None, aux_loss: Optional[bool] = None
) -> deeplabv3.DeepLabV3:
    model = torchvision.models.segmentation.deeplabv3_resnet101(
        weights=None,
        aux_loss=False,
    )
    model.classifier[-1] = torch.nn.Conv2d(
        model.classifier[-1].in_channels,
        1,
        kernel_size=model.classifier[-1].kernel_size,
    )  # change number of outputs to 1
    return model

    # return torchvision.models.segmentation.deeplabv3_restnet101()
    # backbone = resnet.resnet18(weights=None)
    backbone = resnet50(replace_stride_with_dilation=[False, True, True])

    if num_classes is None:
        num_classes = 21

    model = _deeplabv3_resnet(backbone, num_classes, aux_loss)

    model.classifier[-1] = torch.nn.Conv2d(
        model.classifier[-1].in_channels,
        1,
        kernel_size=model.classifier[-1].kernel_size,
    )  # change number of outputs to 1
    return model


def resnet50(**kwargs: Any) -> resnet.ResNet:
    model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], **kwargs)
    # model = resnet.ResNet(resnet.Bottleneck, [1, 2, 3, 1], **kwargs)
    return model


def _deeplabv3_resnet(
    backbone: resnet.ResNet,
    num_classes: int,
    aux: Optional[bool],
) -> deeplabv3.DeepLabV3:
    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(1024, num_classes) if aux else None
    classifier = deeplabv3.DeepLabHead(2048, num_classes)
    return deeplabv3.DeepLabV3(backbone, classifier, aux_classifier)
