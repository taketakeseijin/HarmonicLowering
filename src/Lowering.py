"""
Harmonic Lowering
author: Hirotoshi Takeuchi
"""
import warnings

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HarmonicLowering", "LogHarmonicLowering"]

class HarmonicLowering(nn.Module):
    """
    Lowering input for normal convolution
    """

    def __init__(
        self,
        anchor,
        f_kernel_size,
        in_channels,
        groups=1,
        method="linear",
        ):
        super().__init__()
        self.anchor = anchor
        self.f_kernel_size = f_kernel_size
        self.in_channels = in_channels
        self.groups = groups
        self.method = method

        # setting index rules of in channels
        self.channel_slice_func = lambda k_f: slice(
            k_f*self.in_channels, (k_f+1)*self.in_channels)
        if self.groups != 1:
            warnings.warn(
                "Harmonic Lowering implementation can be not good at group convolution")
            self.channel_slice_func = lambda k_f: slice(
                k_f, None, self.in_channels)

        if self.anchor == 1:
            self.using_func = self.stride_method
            self.method_name = "stride"
        else:
            interp_methods = dict(
                linear=self.linear_method,
            )
            self.using_func = interp_methods[self.method]
            self.method_name = f"interp_{self.mode}"

    def stride_method(self, input, lowered_input):
        """
        Check anchor = 1. If not, you should not use this method.
        return None
        """
        assert input.device() == lowered_input.device(), f"Expected b and A to be on the same device, but found b on {input.device()} and A on {lowered_input.device()} instead."

        if input.is_cuda:
            self._stride_method_gpu(input, lowered_input)
        else:
            self._stride_method_cpu(input, lowered_input)

    def _stride_method_gpu(self, input, lowered_input):
        in_freq_size = input.shape[2]

        current_stream = torch.cuda.current_stream()
        parallel_streams = [torch.cuda.Stream() for _ in range(self.f_kernel_size)]

        # block current stream
        current_stream.synchronize()
        for f in range(self.f_kernel_size):
            # start parallel streams
            with torch.cuda.stream(parallel_streams[f]):
                if f == 0:
                    lowered_input[:, self.channel_slice_func(f), :, :] = input
                else:
                    # get by stride
                    fth_input = input[:, :, ::(f+1), :]
                    # 0-padding
                    pad_shape = (0, 0, 0, in_freq_size - fth_input.shape[2])
                    fth_input = F.pad(fth_input, pad_shape, "constant", 0)
                    # set input
                    lowered_input[:, self.channel_slice_func(f)] = fth_input
        # block parallel streams
        for f in range(self.f_kernel_size):
            parallel_streams[f].synchronize()

    def _stride_method_cpu(self, input, lowered_input):
        in_freq_size = input.shape[2]

        for f in range(self.f_kernel_size):
            if f == 0:
                lowered_input[:, self.channel_slice_func(f), :, :] = input
            else:
                # get by stride
                fth_input = input[:, :, ::(f+1), :]
                # 0-padding
                pad_shape = (0, 0, 0, in_freq_size - fth_input.shape[2])
                fth_input = F.pad(fth_input, pad_shape, "constant", 0)
                # set input
                lowered_input[:, self.channel_slice_func(f)] = fth_input

    def linear_method(self, input, lowered_input):
        assert input.device() == lowered_input.device(), f"Expected b and A to be on the same device, but found b on {input.device()} and A on {lowered_input.device()} instead."

        if input.is_cuda:
            self._linear_method_gpu(input, lowered_input)
        else:
            self._linear_method_cpu(input, lowered_input)

    def _linear_method_gpu(self, input, lowered_input):
        current_stream = torch.cuda.current_stream()
        parallel_streams = [torch.cuda.Stream() for _ in range(self.f_kernel_size)]

        # block current stream
        current_stream.synchronize()
        for f in range(self.f_kernel_size):
            # start parallel streams
            with torch.cuda.stream(parallel_streams[f]):
                lowered_input[:, self.channel_slice_func(f)] = self._get_linear(input, k=f, n=self.anchor)
        # block parallel streams
        for f in range(self.f_kernel_size):
            parallel_streams[f].synchronize()

    def _linear_method_cpu(self, input, lowered_input):
        for f in range(self.f_kernel_size):
            lowered_input[:, self.channel_slice_func(f)] = self._get_linear(input, k=f, n=self.anchor)

    @staticmethod
    def _get_linear(input, k, n):
        """
        return Tensor (input.shape)
        """
        # get X[w*k/n] elements
        _, _, in_freq_size, in_time_size = input.shape

        if k == n:
            return input

        if n == 1:
            strided_input = input[:, :, ::k, :]
            _, _, strided_freq_size, _ = strided_input.shape
            pad_shape = (0, 0, 0, in_freq_size-strided_freq_size)
            return F.pad(strided_input, pad_shape, "constant", 0)

        if k > n:
            trim_freq_size = math.ceil((in_freq_size-1)*k/n)
            trim_freq_size += -trim_freq_size % k + 1
            if trim_freq_size > in_freq_size:
                pad_shape = (0, 0, 0, trim_freq_size-in_freq_size)
                input = F.pad(input, pad_shape, "constant", 0)
            expanded_freq_size = int((trim_freq_size-1)*n/k + 1)
            expanded_input = torch.nn.functional.interpolate(input[:, :, :trim_freq_size, :], size=(
                expanded_freq_size, in_time_size), mode="bilinear", align_corners=True)
            return expanded_input[:, :, :in_freq_size, :]

        if k < n:
            trim_freq_size = math.ceil((in_freq_size-1)*k/n)
            trim_freq_size += -trim_freq_size % k + 1
            if trim_freq_size > in_freq_size:
                pad_shape = (0, 0, 0, trim_freq_size-in_freq_size)
                input = F.pad(input, pad_shape, "constant", 0)
            expanded_freq_size = int((trim_freq_size-1)*n/k + 1)
            expanded_input = torch.nn.functional.interpolate(input[:, :, :trim_freq_size, :], size=(
                expanded_freq_size, in_time_size), mode="bilinear", align_corners=True)
            return expanded_input[:, :, :in_freq_size, :]

        raise Exception("unknown error")

    def forward(self, input):
        """
        input Tensor size(batch,in_channels,f,t)
        lowered_in_channels = in_channels * f_kernel_size
        return Tensor size(batch,lowered_in_channels,f,t)
        """
        # make empty lowered input
        batch, in_channels, in_freq_size, in_time_size = input.shape
        lowered_in_channels = in_channels * self.f_kernel_size
        lowered_shape = (batch, lowered_in_channels,
                         in_freq_size, in_time_size)
        lowered_input = torch.empty(
            lowered_shape, dtype=input.dtype, device=input.device)
        # fill elements
        self.using_func(input, lowered_input)
        return lowered_input

    def extra_repr(self):
        return f"n={self.anchor}, K_f={self.f_kernel_size}, method={self.method_name}"


class LogHarmonicLowering(nn.Module):
    def __init__(
        self,
        anchor,
        f_kernel_size,
        in_channels,
        groups=1,
        out_log_scale=1000,
        in_log_scale=0.001,
        radix=None,
        method="grid",
        ):
        super().__init__()
        self.anchor = anchor
        self.f_kernel_size = f_kernel_size
        self.in_channels = in_channels
        self.groups = groups
        self.out_log_scale = out_log_scale
        self.in_log_scale = in_log_scale
        self.radix = radix
        self.method = method

        # setting index rules of in channels
        self.channel_slice_func = lambda k_f: slice(
            k_f*self.in_channels, (k_f+1)*self.in_channels)
        if self.groups != 1:
            warnings.warn(
                "Harmonic Lowering implementation can be not good at group convolution")
            self.channel_slice_func = lambda k_f: slice(
                k_f, None, self.in_channels)

        methods = dict(
            grid=self.grid_method,
            grid3=self.grid3_method,
            add=self.add_method,
        )
        method_names = dict(
            grid="instant_grid",
            grid3="whole_grid3",
            add="instant_add",
        )
        # grid method is faster
        self.using_func = methods[self.method]
        self.method_name = method_names[self.method]
        if self.method in ["grid", "grid3"]:
            self.grids = None
            self.last_shape = None
        if self.method == "add":
            self.log_shift = self.make_log_shift()    

    def make_log_shift(self):
        """
        compute log shift
        return ndarray size(f_kernel_size)
        """
        assert 1 <= self.anchor <= self.f_kernel_size, f"invalid anchor={self.anchor}. anchor should be in [min=1,f_kernel_size={self.f_kernel_size}]"

        np_shift = (np.arange(self.f_kernel_size) + 1)/self.anchor
        if self.radix is None:
            log_shift = self.out_log_scale * np.log(self.in_log_scale * np_shift)
        else:
            log_shift = self.out_log_scale * \
                np.log(self.in_log_scale * np_shift) / np.log(self.radix)
        target_index = self.anchor - 1
        log_shift -= log_shift[target_index]
        return log_shift

    def make_log_grid(self, input_shape):
        """
        compute grid for grid sample
        return Tensor size(f_kernel_size,batch,f,t,2)
        """
        log_shift = self.make_log_shift()
        batch, _, f, t = input_shape
        grid = torch.empty(self.f_kernel_size, batch, f, t, 2)

        theta = torch.zeros(self.f_kernel_size, batch, 2, 3)
        theta[:, :, np.arange(2), np.arange(2)] = 1
        for fk in range(self.f_kernel_size):
            theta[fk, :, 1, 2] = log_shift[fk]*2/(f-1)
            grid[fk] = F.affine_grid(theta[fk], input_shape)
        return grid

    def make_log_grid3(self, input_shape):
        """
        compute grid for grid sample 3d
        return Tensor size(batch,1,lc,f,t,3)
        """
        grid = self.make_log_grid(input_shape)
        grid3 = torch.zeros(*input_shape,3).unsqueeze(1)
        for f in range(self.f_kernel_size):
            grid3[:, 0, self.channel_slice_func(f), :, 1:] = grid[f].unsqueeze(1)
        return grid3

    def grid_method(self, input, lowered_input):
        """
        Use grid sampler 2d for lowering
        return None
        """
        if self.grids is None or input.shape != self.last_shape:
            # initializing
            self.last_shape = input.shape
            self.grids = torch.as_tensor(
                self.make_log_grid(self.last_shape),
                input.dtype,
                input.device,
                )
        
        for f in range(self.f_kernel_size):
            lowered_input[:, self.channel_slice_func(f)] = F.grid_sample(input, self.grids[f])
    
    def grid3_method(self, input, lowered_input):
        """
        Use grid sampler 3d for lowering.
        This method is done without for loop
        return None
        """
        if self.grids is None or input.shape != self.last_shape:
            # initializing
            self.last_shape = input.shape
            self.grids = torch.as_tensor(
                self.make_log_grid3(self.last_shape),
                input.dtype,
                input.device,
                )

        lowered_input[...] = F.grid_sample(input.unsqueeze(1), self.grids).squeeze(1)
        # Do not replace the above line to lowered_input = ~
        # The object id will change if you do that.

    def add_method(self, input, lowered_input):
        """
        Compute shifted input by adding with weight
        return None
        """
        for f in range(self.f_kernel_size):
            lowered_input[:, self.channel_slice_func(f)] = self.add_shift(input,self.log_shift[f])

    def add_shift(input, log_shift):
        """
        shift
        return Tensor size()
        """
        if log_shift == 0:
            return input
        elif log_shift > 0:
            lf = math.floor(log_shift)
            lc = lf + 1
            wf = lc - log_shift
            wc = 1 - wf
            return wf*F.pad(input[:,:,lf:,:],(0,0,0,lf)) + wc*F.pad(input[:,:,lf+1:,:],(0,0,0,lf+1))
        else:
            lf = math.floor(log_shift)
            lc = lf + 1
            wf = lc - log_shift
            wc = 1 - wf
            return wf*F.pad(input[:,:,lf:,:],(0,0,-lf)) + wc*F.pad(input[:,:,lf-1:,:],(0,0,-lf+1))

    def forward(self, input):
        """
        input Tensor size(batch,in_channels,f,t)
        lowered_in_channels = in_channels * f_kernel_size
        return Tensor size(batch,lowered_in_channels,f,t)
        """
        # make empty lowered input
        batch, in_channels, in_freq_size, in_time_size = input.shape
        lowered_in_channels = in_channels * self.f_kernel_size
        lowered_shape = (batch, lowered_in_channels,
                         in_freq_size, in_time_size)
        lowered_input = torch.empty(
            lowered_shape, dtype=input.dtype, device=input.device)
        # fill elements
        self.using_func(input, lowered_input)
        return lowered_input

    def extra_repr(self):
        radix = self.radix if self.radix is not None else "e"
        return f"n={self.anchor}, K_f={self.f_kernel_size}, method={self.method_name}"\
            + f", log_func(f)={self.out_log_scale}log_{radix} {self.in_log_scale}f"
