import math

import torch
import torch.nn as nn

from STN.modules.gridgen import AffineGridGen
from STN.modules.stn import STN



class Transformer(nn.Module):
    def __init__(self, w, h):
        super(Transformer, self).__init__()
        self.s = STN()
        self.g = AffineGridGen(w, h, lr=0.01)

    def forward(self, input1, input2):
        out = self.g(input2)
        out2 = self.s(input1, out)
        return out2


class STNet(nn.Module):
    def __init__(self, transformer=None):
        super(STNet, self).__init__()
        if transformer is None:
            self.chk_st = False
        else:
            self.chk_st = True

        self.localisation_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )

        self.localisation_fc = nn.Sequential(
            nn.Linear(32 * 32 * 32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6)
        )

        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(48, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 160, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(160, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(192, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )

        self.linear = nn.Sequential(
            nn.Linear(4 * 4 * 192, 3072),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(3072, 3072),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(3072, 3072),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.classification = [nn.Sequential(
            nn.Linear(3072, 11),
            nn.Dropout(0.5)
        ) for _ in range(5)]

        self._initialize_weights()
        self.transformer = transformer

    def forward(self, x):
        fmap = None
        if self.chk_st:
            hidden_local = self.localisation_conv(x.permute(0, 3, 1, 2))
            hidden_local = hidden_local.view(-1, 32 * 32 * 32)
            theta = self.localisation_fc(hidden_local)
            theta = theta.view(-1, 2, 3)
            ch_x = self.transformer(x, theta)
            fmap = self.features(ch_x.permute(0, 3, 1, 2))
        else:
            fmap = self.features(x.permute(0, 3, 1, 2))
        fmap = fmap.view(-1, 4 * 4 * 192)
        hidden = self.linear(fmap)
        val = []
        for i in range(5):
            val.append(self.classification[i](hidden))

        return torch.stack(val, dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
