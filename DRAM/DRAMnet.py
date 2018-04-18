import math

import torch
import torch.nn as nn


class DRAMnet(nn.Module):
    def __init__(self):
        super(DRAMnet, self).__init__()
        self.glimpse_image_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.glimpse_image_fc = nn.Sequential(
            nn.Linear(1024, 1024)
        )

        self.glimpse_loc = nn.Sequential(
            nn.Linear(2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024)
        )

        self.recurrent_1 = nn.LSTMCell(1024, 512)
        self.recurrent_2 = nn.LSTMCell(512, 512)

        self.emission = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2)
        )

        self.context = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.classification = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 10)
        )

        self._initialize_weights()
        self.glimpse_num = 3

    def forward(self, x):
        context = self.context(x)
        emit = self.emission(context.veiw(context.size()[0], -1))
        glimpse_size = [7, 15]
        output = []
        for i in range(6):
            glim = []
            for size in glimpse_size:
                ksize = size * 2 / 5
                if size * 2 % 5 != 0:
                    ksize += 1
                pool = nn.MaxPool2d(kernel_size=ksize, stride=ksize)
                glim.append(pool(x[:, :, emit[0]-size:emit[0]+size, emit[1]-size:emit[1]+size]))
            glim = torch.stack(glim, dim=1)
            Gimg = self.glimpse_image_conv(glim)
            Gimg = self.glimpse_image_fc(Gimg)
            Gloc = self.glimpse_loc(emit)

            glimpse = torch.mul(Gimg, Gloc)

            rn1 = self.recurrent_1(glimpse)
            rn2 = self.recurrent_2(rn1)

            output += self.classification(rn1)
            emit = self.emission(rn2)

        return output

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
