
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def make_log_shift(f_kernel_size, n_anchor=1, out_log_scale=1000, in_log_scale=0.001, radix=None):
    assert 1 <= n_anchor <= f_kernel_size, f"invalid n_anchor={n_anchor}. n_anchor should be in [min=1,f_kernel_size={f_kernel_size}]"

    np_shift = np.arange(f_kernel_size) + 1
    if radix is None:
        log_shift = out_log_scale * np.log(in_log_scale * np_shift)
    else:
        log_shift = out_log_scale * \
            np.log(in_log_scale * np_shift) / np.log(radix)
    target_index = n_anchor - 1
    log_shift -= log_shift[target_index]
    return log_shift


def make_grid(log_shift, input_shape):
    f_kernel_size = len(log_shift)
    batch, _, f, t = input_shape
    grid = torch.empty(f_kernel_size, batch, f, t, 2)

    theta = torch.zeros(f_kernel_size, batch, 2, 3)
    theta[:, :, 0, 0] = 1
    theta[:, :, 1, 1] = 1
    for fk in range(f_kernel_size):
        theta[fk, :, 1, 2] = log_shift[fk]*2/(f-1)
        grid[fk] = torch.nn.functional.affine_grid(theta[fk], input_shape)
    return grid


class LogSingleHarmonicConv2d(nn.Conv2d):
    def __init__(self, *args, n_anchor=1, out_log_scale=1000, in_log_scale=0.001, radix=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_anchor = n_anchor
        # out_log_scale * log_radix (in_log_scale*x)
        # None treated as e
        self.out_log_scale = out_log_scale
        self.in_log_scale = in_log_scale
        self.radix = radix
        self.log_shift = make_log_shift(
            self.kernel_size[0], self.n_anchor, self.out_log_scale, self.in_log_scale, self.radix)
        self.grids = None
        self.last_shape = None

        self.reformed_weight = torch.nn.Parameter(self.weight.transpose(1, 2).contiguous().view(
            self.out_channels, self.in_channels*self.kernel_size[0], 1, self.kernel_size[1]))
        self.weight = None

    def get_weight(self):
        return self.reformed_weight.view(self.out_channels, self.kernel_size[0], self.in_channels, self.kernel_size[1]).transpose(1, 2)

    def set_weight(self, weight):
        outc, inc, kf, kt = weight.shape
        if outc != self.out_channels or inc != self.in_channels or (kf, kt) != self.kernel_size:
            raise Exception("weight shape is incorrect")
        weight.device = self.reformed_weight.device
        weight.dtype = self.reformed_weight.dtype
        self.reformed_weight = torch.nn.Parameter(weight.transpose(1, 2).contiguous().view(
            self.out_channels, self.in_channels*self.kernel_size[0], 1, self.kernel_size[1]))

    @staticmethod
    def functional(input, reformed_weight, grids=None, out_log_scale=1000, in_log_scale=0.001, radix=None, n_anchor=1, bias=None, stride=1, padding=0, dilation=1, groups=1):
        batch, in_channels, in_freq_size, in_time_size = input.shape
        _, expanded_in_channels, _, _ = reformed_weight.shape
        f_kernel_size = int(expanded_in_channels / in_channels)
        stride = _pair(stride)
        padding = (0, _pair(padding)[1])
        dilation = _pair(dilation)

        assert 1 <= n_anchor <= f_kernel_size, f"invalid n_anchor={n_anchor}. n_anchor should be in [min=1,f_kernel_size={f_kernel_size}]"

        if grids is None:
            log_shift = make_log_shift(
                f_kernel_size, n_anchor, out_log_scale, in_log_scale, radix)
            grids = make_grid(log_shift, input.shape)
        if groups != 1:
            raise NotImplementedError

        expanded_2d_input = torch.empty(
            batch, expanded_in_channels, in_freq_size, in_time_size, device=input.device)

        for f in range(f_kernel_size):
            expanded_2d_input[:, f*in_channels:(f+1)*in_channels, :, :] = torch.nn.functional.grid_sample(
                input, grids[f], mode="bilinear", padding_mode="zeros")

        output = F.conv2d(
            input=expanded_2d_input,
            weight=reformed_weight,
            bias=bias, stride=stride, padding=padding,
            dilation=dilation, groups=groups)

        return output

    def forward(self, input):
        if self.grids is None or input.shape != self.last_shape:
            self.last_shape = input.shape
            self.grids = make_grid(
                self.log_shift, self.last_shape).to(input.device)
        output = self.functional(input, self.reformed_weight, self.grids, self.out_log_scale, self.in_log_scale, self.radix, self.n_anchor, self.bias,
                                 self.stride, self.padding, self.dilation, self.groups)
        return output

    def extra_repr(self):
        return f"out_log_scale={self.out_log_scale}, in_log_scale={self.in_log_scale}, radix={self.radix}, " + super().extra_repr()
