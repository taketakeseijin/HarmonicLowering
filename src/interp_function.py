import torch
import torch.nn.functional as F
import interp_same as IS

def shift_minus(input,shift):
    fliped_input = torch.flip(input,(3,)).contiguous()
    fliped_output = torch.empty_like(input)
    IS.interp_shift_plus(fliped_input,fliped_output,-shift)
    output = torch.flip(fliped_output,(3,)).contiguous()
    return output

class Shift(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input,shift):
        assert input.dim() == 4, "this method suppose the dimension of input is 4"
        ctx.shift = shift

        if shift == 0:
            # special case
            output = input
        elif shift > 0:
            output = torch.empty_like(input)
            IS.interp_shift_plus(input,output,shift)
        else:
            output = shift_minus(input,shift)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        shift = -ctx.shift
        grad_input = grad_shift = None
        if ctx.needs_input_grad[0]:
            if shift == 0:
                grad_input = grad_output
            elif shift > 0:
                grad_input = torch.empty_like(grad_output)
                IS.interp_shift_plus(grad_output,grad_input,-shift)
            else:
                grad_input = shift_minus(grad_output,shift)
        return grad_input,grad_shift

ShiftFunctional = Shift.apply

def make_detail(k,n,device):
    index = torch.arange(n,dtype=torch.int32,device=device) * k / n 
    weight = 1 - (torch.arange(n,dtype=torch.float32,device=device) * k / n - index)
    return index, weight

class ZoomF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k, n):
        ctx.k = k
        ctx.n = n
        output = torch.empty_like(input)
        indexes, weights = make_detail(k,n,device=input.device)
        IS.interp_affine(input,output,indexes,weights,k,n)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_k = grad_n = None
        k = ctx.n
        n = ctx.k
        if ctx.needs_input_grad[0]:
            if n == 1:
                # stride case
                # [batch,channel,time,freq]
                freq_size = grad_output.shape[3]
                # get by stride
                strided_grad_output = grad_output[:, :, :, ::k]
                # 0-padding
                pad_shape = (0, freq_size - strided_grad_output.shape[3])
                grad_input = F.pad(strided_grad_output, pad_shape, "constant", 0)
            else:
                grad_input = torch.empty_like(grad_output)
                indexes, weights = make_detail(k,n,device=grad_output.device)
                IS.interp_affine(grad_output,grad_input,indexes,weights,k,n)
        return grad_input, grad_k, grad_n


ZoomFunctional = ZoomF.apply

class Zoom(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, k, n):
        # special case
        if n == k:
            return input
        # stride case
        if n == 1:
            # [batch,channel,time,freq]
            freq_size = input.shape[3]
            # get by stride
            strided_input = input[:, :, :, ::k]
            # 0-padding
            pad_shape = (0, freq_size - strided_input.shape[3])
            return F.pad(strided_input, pad_shape, "constant", 0)
        
        # interpolate same size by affine 1d interpolation
        return ZoomFunctional(input, k, n)

