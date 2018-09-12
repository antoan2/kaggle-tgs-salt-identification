import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet


class NullMaskClassifier(nn.Module):
    def __init__(self):
        super(NullMaskClassifier, self).__init__()

        # base = resnet.resnet18(pretrained=True)
        base = resnet.resnet18()
        base.load_state_dict(torch.load('/models/resnet18-5c106cde.pth'))
        conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        conv1.weight.data = base.conv1.weight[:, 0, ...].resize(64, 1, 7, 7)
        self.maxpool = base.maxpool

        self.in_block = nn.Sequential(conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=512, out_features=2, bias=True)

        # self.fc = base.fc

    def forward(self, x):
        # Initial block
        x = self.in_block(x)

        # Encoder blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)


        return x

    def get_predictions(self, outputs):
        values, indices = outputs.max(1)
        return indices
