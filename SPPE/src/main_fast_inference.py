import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import numpy as np
from SPPE.src.utils.img import flip, shuffleLR
from SPPE.src.utils.eval import getPrediction
from SPPE.src.models.FastPose import FastPose

import time
import sys

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


class InferenNet(nn.Module):
    def __init__(self, dataset, weights_file='./Models/sppe/fast_res101_320x256.pth'):
        super().__init__()

        self.pyranet = FastPose('resnet101').cuda()
        print('Loading pose model from {}'.format(weights_file))
        sys.stdout.flush()
        self.pyranet.load_state_dict(torch.load(weights_file))
        self.pyranet.eval()
        self.pyranet = model

        self.dataset = dataset

    def forward(self, x):
        out = self.pyranet(x)
        out = out.narrow(1, 0, 17)

        flip_out = self.pyranet(flip(x))
        flip_out = flip_out.narrow(1, 0, 17)

        flip_out = flip(shuffleLR(
            flip_out, self.dataset))

        out = (flip_out + out) / 2

        return out


class InferenNet_fast(nn.Module):
    def __init__(self, weights_file='./Models/sppe/fast_res101_320x256.pth'):
        super().__init__()

        self.pyranet = FastPose('resnet101').cuda()
        print('Loading pose model from {}'.format(weights_file))
        self.pyranet.load_state_dict(torch.load(weights_file))
        self.pyranet.eval()

    def forward(self, x):
        out = self.pyranet(x)
        out = out.narrow(1, 0, 17)

        return out


class InferenNet_fastRes50(nn.Module):
    def __init__(self, weights_file='./Models/sppe/fast_res50_256x192.pth'):
        super().__init__()

        self.pyranet = FastPose('resnet50', 17).cuda()
        print('Loading pose model from {}'.format(weights_file))
        self.pyranet.load_state_dict(torch.load(weights_file))
        self.pyranet.eval()

    def forward(self, x):
        out = self.pyranet(x)

        return out
