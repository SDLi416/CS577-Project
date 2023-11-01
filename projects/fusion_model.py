import torchvision
import torch.nn as nn
from echonet.projects.unet_plus_plus import UnetPlusPlus
from torchvision.models import segmentation

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.deeplab = segmentation.deeplabv3_resnet101(weights=segmentation.DeepLabV3_ResNet101_Weights.DEFAULT, aux_loss=True)
        self.backbone = self.deeplab.backbone
        self.unetplusplus = UnetPlusPlus(1)
        self.classifier = self.deeplab.classifier
        self.channel_reducer = nn.Conv2d(2048, 512, 1)  # Channel reducer
        self.final_upsample = nn.Upsample(size=(112, 112), mode='bilinear', align_corners=False)
        self.channel_adjust = nn.Conv2d(1, 2048, 1)

    def forward(self, x):
        feature_map = self.backbone(x)
        feature_map = feature_map['out']
        feature_map = self.channel_reducer(feature_map)  # Reducing channels
        feature_map = self.unetplusplus(feature_map)
        feature_map = self.channel_adjust(feature_map)
        output = self.classifier(feature_map)
        output = self.final_upsample(output)
        output = {'out': output}
        return output

def main():
    model = FusionModel()
    print(model)

if __name__ == '__main__':
    main()
