import torch
import torch.nn as nn

class ContinuousParallelConv(nn.Module):
    def __init__(self, in_channels, out_channels, pre_Batch_Norm=True):
        super(ContinuousParallelConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if pre_Batch_Norm:
            self.Conv_forward = nn.Sequential(
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(),
                nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))
        else:
            self.Conv_forward = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU())

    def forward(self, x):
        x = self.Conv_forward(x)
        return x

class UnetPlusPlus(nn.Module):
    def __init__(self, num_classes):
        super(UnetPlusPlus, self).__init__()
        self.num_classes = num_classes
        self.channels = 512

        self.CONV2_1 = ContinuousParallelConv(self.channels * 2, self.channels, pre_Batch_Norm=True)

        self.CONV1_1 = ContinuousParallelConv(self.channels * 2, self.channels, pre_Batch_Norm=True)
        self.CONV1_2 = ContinuousParallelConv(self.channels * 3, self.channels, pre_Batch_Norm=True)

        self.CONV0_1 = ContinuousParallelConv(self.channels * 2, self.channels, pre_Batch_Norm=True)
        self.CONV0_2 = ContinuousParallelConv(self.channels * 3, self.channels, pre_Batch_Norm=True)
        self.CONV0_3 = ContinuousParallelConv(self.channels * 4, self.channels, pre_Batch_Norm=True)

        self.stage_0 = ContinuousParallelConv(self.channels, self.channels, pre_Batch_Norm=False)
        self.stage_1 = ContinuousParallelConv(self.channels, self.channels, pre_Batch_Norm=False)
        self.stage_2 = ContinuousParallelConv(self.channels, self.channels, pre_Batch_Norm=False)

        self.pool = nn.MaxPool2d(2)

        self.upsample_2_1 = nn.ConvTranspose2d(self.channels, self.channels, 4, stride=2, padding=1) 

        self.upsample_1_1 = nn.ConvTranspose2d(self.channels, self.channels, 4, stride=2, padding=1, output_padding=1)
        self.upsample_1_2 = nn.ConvTranspose2d(self.channels, self.channels, 4, stride=2, padding=1, output_padding=1) 

        self.upsample_0_1 = nn.ConvTranspose2d(self.channels, self.channels, 4, stride=2, padding=1) 
        self.upsample_0_2 = nn.ConvTranspose2d(self.channels, self.channels, 4, stride=2, padding=1) 
        self.upsample_0_3 = nn.ConvTranspose2d(self.channels, self.channels, 4, stride=2, padding=1) 

        self.final = nn.Sequential(
            nn.Conv2d(self.channels, self.num_classes, 1)
        )

    def forward(self, x):
        x_0_0 = self.stage_0(x)
        x_1_0 = self.stage_1(self.pool(x_0_0))
        x_2_0 = self.stage_2(self.pool(x_1_0))

        x_0_1 = torch.cat([self.upsample_0_1(x_1_0), x_0_0], 1)
        x_0_1 = self.CONV0_1(x_0_1)

        x_1_1 = torch.cat([self.upsample_1_1(x_2_0), x_1_0], 1)
        x_1_1 = self.CONV1_1(x_1_1)

        x_0_2 = torch.cat([self.upsample_0_2(x_1_1), x_0_0, x_0_1], 1)
        x_0_2 = self.CONV0_2(x_0_2)

        x_1_2 = torch.cat([self.upsample_1_2(x_2_0), x_1_0, x_1_1], 1)
        x_1_2 = self.CONV1_2(x_1_2)

        x_0_3 = torch.cat([self.upsample_0_3(x_1_2), x_0_0, x_0_1, x_0_2], 1)
        x_0_3 = self.CONV0_3(x_0_3)

        output = self.final(x_0_3)
        return output