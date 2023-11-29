import torch.nn as nn
from torch.nn import functional as F
import torch
from echonet.projects.unet_plus_plus import UnetPlusPlus
from torchvision.models import segmentation
from echonet.projects.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from echonet.projects.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.deeplab = segmentation.deeplabv3_resnet50(aux_loss=True).double().cuda()
        self.unetplusplus = UnetPlusPlus(num_classes=1).cuda()

        vit_name = 'R50-ViT-B_16'

        config_vit = CONFIGS_ViT_seg[vit_name]
        config_vit.n_classes = 1
        img_size = 112
        if vit_name.find('R50') != -1:
                config_vit.patches.grid = (int(img_size / 16), int(img_size / 16))

        self.trans_unet = ViT_seg(config_vit, img_size, num_classes=config_vit.n_classes).double().cuda()
        self.classifier = self.deeplab.classifier
         # Convolutional layers for merging
        self.conv1 = nn.Conv2d(2, 128, kernel_size=3, padding=1)  # Adjusted number of input channels
        self.conv2 = nn.Conv2d(128, 1, kernel_size=3, padding=1)  # Adjusted number of input channels

    def forward(self, x):
        deeplab_out = self.deeplab(x)['out']
        unet_out = self.unetplusplus(x)
        vit_out = self.trans_unet(x)
        out = self.merge_output(deeplab_out, vit_out)
        result = {'out': out}
        return result
    
    def merge_output(self, deeplab_out, unet_out):
       return (deeplab_out + unet_out) / 2
    
    # def merge_output(self, deeplab_out, unet_out):
    #     # Concatenating the outputs along the channel dimension
    #     merged = torch.cat((deeplab_out, unet_out), dim=1)

    #     # Passing through convolutional layers
    #     merged = F.leaky_relu(self.conv1(merged))
    #     merged = F.leaky_relu(self.conv2(merged))
    #     print(merged)
        
    #     return merged

def main():
    model = FusionModel()
    print(model)

if __name__ == '__main__':
    main()
