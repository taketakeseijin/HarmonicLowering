"""
Harmonic Convolution implementation by Harmonic Lowering in pytorch
author: Hirotoshi Takeuchi
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from Lowering import HarmonicLowering
from Lowering import LogHarmonicLowering

__all__ = ["SingleHarmonicConv2d", "SingleLogHarmonicConv2d"]

class BaseSingleHarmonicConv2d(nn.Conv2d):
    """
    Base class for Harmonic Convolution
    """
    
    def __init__(self, *args, anchor=1, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(anchor, int):
            raise Exception("anchor should be integer")
        self.anchor = anchor

        if self.anchor < 1:
            raise Exception("anchor should be equal to or bigger than 1")
        if self.padding_mode != "zeros":
            raise NotImplementedError("only zero padding mode is implemented")

        if self.padding[0] != 0:
            warnings.warn(
                "Harmonic Convolution do no padding on frequency axis")
            self.padding[0] = 0

        # transforming weight shape
        self.lowered_weight = None
        self.__set_weight(self.weight)
        self.weight = None

    def get_weight(self):
        """
        get lowered weight with normal weight shape
        return Tensor (C_out, C_in, K_f, K_t)
        """
        return self.lowered_weight.view(
            self.out_channels,
            self.kernel_size[0],
            int(self.in_channels/self.groups),
            self.kernel_size[1],
        ).transpose(1, 2)

    def set_weight(self, weight):
        """
        set weight on lowered weight with checking
        return None
        """
        expected_shape = (self.out_channels, int(
            self.in_channels/self.groups), *self.kernel_size)
        if weight.shape != expected_shape:
            raise Exception("weight shape is incorrect")
        self.__set_weight(torch.as_tensor(
            weight,
            dtype=self.lowered_weight.dtype,
            device=self.lowered_weight.device
        ))

    def __set_weight(self, weight):
        """
        set lowered weight without checking
        return None
        """
        self.lowered_weight = torch.nn.Parameter(self.transform_weight(weight))

    @staticmethod
    def transform_weight(weight):
        """
        transform weight to lowered weight
        return Tensor of lowered weight
        """
        out_channels, in_channels, k_f, k_t = weight.shape
        lowered_shape = (out_channels, in_channels*k_f, 1, k_t)
        return weight.transpose(1, 2).contiguous().view(lowered_shape)

    def forward(self, input):
        raise NotImplementedError("overwrite forward method")


class SingleHarmonicConv2d(BaseSingleHarmonicConv2d):
    """
    Harmonic Convolution by Harmonic Lowering
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.HL = HarmonicLowering(
            anchor=self.anchor,
            f_kernel_size=self.kernel_size[0],
            in_channels=self.in_channels,
            groups=self.groups,
            method="linear",
            expand_ahead=True,
        )

    def forward(self, input):
        """
        input Tensor size(batch,in_channels,f,t)
        return Tensor size(batch,out_channels,f',t')
        """
        lowered_input = self.HL(input)

        output = F.conv2d(
            input=lowered_input,
            weight=self.lowered_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return output

    def extra_repr(self):
        return f"anchor={self.anchor}, " + super().extra_repr()


class SingleLogHarmonicConv2d(nn.Conv2d):
    def __init__(self, *args, out_log_scale=1000, in_log_scale=0.001, radix=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.out_log_scale = out_log_scale
        self.in_log_scale = in_log_scale
        self.radix = radix

        self.LHL = LogHarmonicLowering(
            anchor=self.anchor,
            f_kernel_size=self.kernel_size[0],
            in_channels=self.in_channels,
            groups=self.groups,
            out_log_scale=self.out_log_scale,
            in_log_scale=self.in_log_scale,
            radix=self.radix,
            method="grid",
        )

    def forward(self, input):
        """
        input Tensor size(batch,in_channels,f,t)
        return Tensor size(batch,out_channels,f',t')
        """
        lowered_input = self.LHL(input)

        output = F.conv2d(
            input=lowered_input,
            weight=self.lowered_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return output

    def extra_repr(self):
        return f"out_log_scale={self.out_log_scale}, in_log_scale={self.in_log_scale}, radix={self.radix}, " + super().extra_repr()
