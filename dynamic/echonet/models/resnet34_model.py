import torch.nn as nn

import torchvision


class ResNet34Model(nn.Module):
    def __init__(self):
        super(ResNet34Model, self).__init__()
        self.resnet34 = torchvision.models.resnet34()
        self.line = nn.Linear(1000, 112 * 112)

    def forward(self, x):
        out = self.resnet34(x)
        out = self.line(out)
        out = out.reshape(out.shape[0], 1, 112, 112)
        result = {"out": out}
        return result
