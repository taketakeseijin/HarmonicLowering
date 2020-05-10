
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class SingleHarmonicConv2d(nn.Conv2d):
    def __init__(self, *args, n_anchor=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_anchor = n_anchor

        if self.n_anchor < 1:
            raise Exception("n_anchor should be equal to or bigger than 1")
        if self.groups != 1:
            raise NotImplementedError("grouped convolution is not implemented")
        if self.padding_mode != "zeros":
            raise NotImplementedError("only zero padding mode is implemented")
        if self.padding[0] != 0:
            warnings.warn(
                "Harmonic Convolution do no padding on frequency axis")

        self.reformed_weight = nn.Parameter(self.weight.transpose(1, 2).contiguous().view(
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
    def functional2d(input, reformed_weight, n_anchor=1, bias=None, stride=1, padding=0, dilation=1, groups=1):
        batch, in_channels, in_freq_size, in_time_size = input.shape
        _, expanded_in_channels, _, _ = reformed_weight.shape
        f_kernel_size = int(expanded_in_channels / in_channels)
        stride = _pair(stride)
        padding = (0, _pair(padding)[1])
        dilation = _pair(dilation)
        if groups != 1:
            raise NotImplementedError("grouped convolution is not Implemented")

        # expand input
        if n_anchor > 1:
            Interp_Mode = "linear"
            expand_freq_size = n_anchor * in_freq_size - (n_anchor-1)
            if Interp_Mode == "bilinear":
                expanded_input = F.interpolate(input, size=(expand_freq_size, in_time_size),
                                               mode="bilinear", align_corners=True)
            elif Interp_Mode == "linear":
                expanded_input = F.interpolate(input.view(batch*in_channels, in_freq_size, in_time_size).transpose(1, 2), size=(expand_freq_size),
                                               mode="linear", align_corners=True).view(batch, in_channels, in_time_size, expand_freq_size).transpose(2, 3)
            else:
                raise NotImplementedError
        else:
            # no expand
            expand_freq_size = in_freq_size
            expanded_input = input

        expanded_2d_input = torch.empty(
            batch, expanded_in_channels, in_freq_size, in_time_size, device=input.device)

        for f in range(f_kernel_size):
            fth_input = expanded_input[:, :, ::f+1, :]
            _, _, fth_freq_size, _ = fth_input.shape
            if fth_freq_size < in_freq_size:
                pad_shape = (0, 0, 0, in_freq_size-fth_freq_size)
                fth_input = F.pad(fth_input, pad_shape, "constant", 0)

            expanded_2d_input[:, f*in_channels:(f+1)*in_channels,
                              :, :] = fth_input[:, :, :in_freq_size, :]

        output = F.conv2d(
            input=expanded_2d_input,
            weight=reformed_weight,
            bias=bias, stride=stride, padding=padding,
            dilation=dilation, groups=groups)

        return output

    def forward(self, input):
        output = self.functional2d(
            input=input,
            reformed_weight=self.reformed_weight,
            n_anchor=self.n_anchor,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return output

    def extra_repr(self):
        return f"anchor={self.n_anchor}, " + super().extra_repr()
