import torch.nn as nn

import torchvision


class ResNet18Model(nn.Module):
    def __init__(self):
        super(ResNet18Model, self).__init__()
        self.resnet18 = torchvision.models.resnet18()
        self.line = nn.Linear(1000, 112 * 112)

    def forward(self, x):
        out = self.resnet18(x)
        out = self.line(out)
        out = out.reshape(out.shape[0], 1, 112, 112)
        result = {"out": out}
        return result
