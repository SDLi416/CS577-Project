import torch.nn as nn
from echonet.projects.unet_plus_plus import UnetPlusPlus
from torchvision.models import segmentation

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.deeplab = segmentation.deeplabv3_resnet50(aux_loss=True)
        self.unetplusplus = UnetPlusPlus(num_classes=1)
        self.classifier = self.deeplab.classifier

    def forward(self, x):
        deeplab_out = self.deeplab(x)['out']
        unet_out = self.unetplusplus(x)
        out = self.merge_output(deeplab_out, unet_out)
        result = {'out': out}
        return result
    
    def merge_output(self, deeplab_out, unet_out):
        return (deeplab_out + unet_out) / 2

def main():
    model = FusionModel()
    print(model.backbone['out'].out_channels)

if __name__ == '__main__':
    main()
