import torch
import torch.nn.functional as F
import interp_same as IS


class ShiftF(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input,shift,time_last):
        assert input.dim() == 4, "this method suppose the dimension of input is 4"
        ctx.shift = shift
        ctx.time_last = time_last

        output = torch.empty_like(input)
        IS.interp_shift(input,output,shift,time_last)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        shift = -ctx.shift
        time_last = ctx.time_last
        grad_input = grad_shift = None
        if ctx.needs_input_grad[0]:
            grad_output = grad_output.contiguous()
            grad_input = torch.empty_like(grad_output)
            IS.interp_shift(grad_output,grad_input,shift,time_last)
        return grad_input,grad_shift,None

ShiftFunctional = ShiftF.apply

class Shift(torch.nn.Module):
    def __init__(self,time_last=True):
        super().__init__()
        self.time_last = time_last

    def forward(self,input,shift):
        # input [batch,channel,freq,time]
        if shift == int(shift):
            # special case
            if shift == 0:
                return input
            if self.time_last:
                if shift > 0:
                    return F.pad(input[:,:,:-int(shift),:],(0,0,int(shift),0),"constant",0)
                else:
                    return F.pad(input[:,:,-int(shift):,:],(0,0,0,-int(shift)),"constant",0)
            else:
                if shift > 0:
                    return F.pad(input[:,:,:,:-int(shift)],(int(shift),0),"constant",0)
                else:
                    return F.pad(input[:,:,:,-int(shift):],(0,-int(shift)),"constant",0)
        else:
            return ShiftFunctional(input,shift,self.time_last)
    

def make_detail(k,n,device):
    index = torch.arange(n,dtype=torch.int32,device=device) * k / n 
    weight = 1 - (torch.arange(n,dtype=torch.float32,device=device) * k / n - index)
    return index, weight

class ZoomF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k, n, time_last):
        ctx.k = k
        ctx.n = n
        ctx.time_last = time_last
        output = torch.empty_like(input)
        indexes, weights = make_detail(k,n,device=input.device)
        IS.interp_affine(input,output,indexes,weights,k,n,time_last)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_k = grad_n = None
        k = ctx.n
        n = ctx.k
        time_last = ctx.time_last
        if ctx.needs_input_grad[0]:
            grad_output = grad_output.contiguous()
            if n == 1:
                # stride case
                if time_last:
                    # [batch,channel,freq,time]
                    freq_size = grad_output.shape[2]
                    # get by stride
                    strided_grad_output = grad_output[:, :, ::k, :]
                    # 0-padding
                    pad_shape = (0,0,0, freq_size - strided_grad_output.shape[2])
                    grad_input = F.pad(strided_grad_output, pad_shape, "constant", 0)
                else:
                    freq_size = grad_output.shape[3]
                    # get by stride
                    strided_grad_output = grad_output[:, :, :, ::k]
                    # 0-padding
                    pad_shape = (0, freq_size - strided_grad_output.shape[3])
                    grad_input = F.pad(strided_grad_output, pad_shape, "constant", 0)
            else:
                grad_input = torch.empty_like(grad_output)
                indexes, weights = make_detail(k,n,device=grad_output.device)
                IS.interp_affine(grad_output,grad_input,indexes,weights,k,n,time_last)
        return grad_input, grad_k, grad_n,None


ZoomFunctional = ZoomF.apply

class Zoom(torch.nn.Module):
    def __init__(self,time_last=True):
        super().__init__()
        self.time_last = time_last
    
    def forward(self, input, k, n):
        # special case
        if n == k:
            return input
        # interpolate same size by affine 1d interpolation
        return ZoomFunctional(input, k, n, self.time_last)

