import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import math
from resnet import resnet18, ResDeconv, BasicBlock
import torch.nn.functional as F


class GazeStatic(nn.Module):
    def __init__(self):
        super(GazeStatic, self).__init__()
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame

        self.base_model = resnet18(pretrained=True)

        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)

        # The linear layer that maps the LSTM with the 3 outputs
        self.last_layer = nn.Linear(self.img_feature_dim, 6)
        # self.tanh = nn.Tanh()

    def forward(self, x_in):
        base_out, _, _ = self.base_model(x_in)
        embedding = torch.flatten(base_out, start_dim=1)
        output = self.last_layer(embedding)
        # output = self.tanh(output)

        gaze_output = output[:, :3]
        head_output = output[:, 3:]

        return gaze_output, head_output


class RTRegress(nn.Module):
    def __init__(self):
        super(RTRegress, self).__init__()
        self.inplanes = 1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.inplanes, 500)
        self.fc2 = nn.Linear(500, 3)

    def forward(self, gf1, pf1, gf2, pf2):
        feature = torch.cat((gf1, pf1, gf2, pf2), dim=1)
        x = self.avgpool(feature)
        x = x.view(x.size(0), -1)
        # x = nn.Dropout()(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)

        return x


class AngularLoss():
    def __init__(self):
        pass

    def __call__(self, gaze, label):
        gaze = gaze / torch.norm(gaze, 2, dim=1).reshape(-1, 1)
        label = label / torch.norm(label, 2, dim=1).reshape(-1, 1)
        cos = torch.sum(gaze * label, dim=1)
        cos = torch.minimum(cos, 0.999999 * torch.ones_like(cos))
        angular_loss = torch.mean(torch.arccos(cos))
        return angular_loss


class StableLossTerm():
    def __init__(self):
        pass

    def __call__(self, hp1, hp2):
        it1 = torch.sin(hp1[:, 1]) * torch.sin(hp2[:, 1])
        it2 = torch.cos(hp1[:, 1]) * torch.sin(hp1[:, 0]) * torch.cos(hp2[:, 1]) * torch.sin(hp2[:, 0])
        it3 = torch.cos(hp1[:, 1]) * torch.cos(hp1[:, 0]) * torch.cos(hp2[:, 1]) * torch.cos(hp2[:, 0])
        return torch.mean(it1 + it2 + it3)


class StableLossTerm_w_Rmat():
    def __init__(self):
        pass

    def __call__(self, hp1, hp2, R_mat1, R_mat2):
        column3_1 = torch.cat((torch.sin(hp1[:, 1]).reshape(-1, 1),
                               (torch.cos(hp1[:, 1]) * torch.sin(hp1[:, 0])).reshape(-1, 1),
                               (torch.cos(hp1[:, 1]) * torch.cos(hp1[:, 0])).reshape(-1, 1)), 1)
        column3_2 = torch.cat((torch.sin(hp2[:, 1]).reshape(-1, 1),
                               (torch.cos(hp2[:, 1]) * torch.sin(hp2[:, 0])).reshape(-1, 1),
                               (torch.cos(hp2[:, 1]) * torch.cos(hp2[:, 0])).reshape(-1, 1)), 1)
        # print(column3_2.shape, column3_1.shape)
        col3_1 = torch.bmm(R_mat1.permute(0, 2, 1), column3_1.reshape(-1, 3, 1))
        col3_2 = torch.bmm(R_mat2.permute(0, 2, 1), column3_2.reshape(-1, 3, 1))

        # print(col3_1.shape, col3_2.shape)

        return torch.mean(col3_1 * col3_2)
