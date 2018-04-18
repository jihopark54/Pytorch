import math

import torch
import torch.nn as nn

from DLAN.modules.vgg import vgg16, dilation_pretrained
from DLAN.modules.stn import STN
from DLAN.modules.gridgen import AffineGridGen


class Transformer(nn.Module):
    def __init__(self, w, h):
        super(Transformer, self).__init__()
        self.s = STN()
        self.g = AffineGridGen(w, h, lr=0.01)

    def forward(self, input1, input2):
        out = self.g(input2)
        out2 = self.s(input1, out)
        return out2

#512 , 512
class DLAN(nn.Module):
    def __init__(self, dilate_list, transformer):
        super(DLAN, self).__init__()

        self.trans = transformer

        self.regression = [nn.Sequential(
            nn.Linear(32 * 32 * 512, 32 * 32 * 512),
            nn.ReLU(),
            nn.Linear(32 * 32 * 512, 32 * 32 * 512),
            nn.ReLU(),
            nn.Linear(32 * 32 * 512, 4)
        ) for _ in range(8)]
        self.recurrent = [nn.Sequential(
            nn.Linear(32 * 32 * 512, 32 * 32 * 512),
            nn.ReLU(),
            nn.Linear(32 * 32 * 512, 32 * 32 * 512),
            nn.ReLU(),
            nn.Linear(32 * 32 * 512, 6)
        ) for _ in range(8)]
        self._initialize_weights()
        self.recurrent_num = 3
        self.dilated_features = [dilation_pretrained(vgg16(), d) for d in dilate_list]

    def forward(self, x):

        # Selective Dilated Convolution
        selective = []
        for feature in self.dilated_features:
            selective.append(feature(x))

        selective = torch.stack(selective)
        selective = torch.max(selective)

        # HR Spatial Transformer
        theta = [0 for _ in range(8)]
        trans_conv = [selective for _ in range(8)]

        for _ in range(self.recurrent_num):
            for idx, recurrent in enumerate(self.recurrent):
                changed_theta = recurrent(trans_conv[idx].view(-1 ,32 * 32 * 512))
                trans_conv[idx] = self.trans(trans_conv[idx], changed_theta)
                theta[idx] = theta[idx] * changed_theta

        result = []
        for idx, regression in enumerate(self.regression):
            point = regression(trans_conv[idx].view(-1 ,32 * 32 * 512))
            result.append(theta[idx] * point)

        return torch.stack(result, dim=1)

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
