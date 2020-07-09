#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>

#define CUDA_MAX_THREADS 1024 // this is safe.

namespace{
    template <typename scalar_t>
    __global__ void interp_affine_out_kernel(
        const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> idata,
        torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> odata,
        const torch::PackedTensorAccessor<int,1,torch::RestrictPtrTraits,size_t> index,
        const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> weight,
        const int num_kernels,
        const int k,
        const int n
    ){
        const int batchsize = idata.size(0);
        const int channelsize = idata.size(1);
        const int freqsize = idata.size(2);
        const int timesize = idata.size(3);

        const int id = blockIdx.x * blockDim.x + threadIdx.x;
        const int time_id = id % timesize;
        const int freq_id = id / timesize;

        const int step = freq_id / n;
        const int nn = freq_id % n;

        const int ref_index_0 = step * k + index[nn];
        const int ref_index_1 = ref_index_0 + 1;
        const float weight_0 = weight[nn];
        const float weight_1 = 1 - weight_0;

        if (id < num_kernels && ref_index_1 < freqsize){
            for (int b = 0; b < batchsize; b++){
                for (int c = 0; c < channelsize; c++){
                    odata[b][c][freq_id][time_id] = weight_0 * idata[b][c][ref_index_0][time_id] + weight_1 * idata[b][c][ref_index_1][time_id]; 
                }
            }
        }
    }

    template <typename scalar_t>
    __global__ void interp_shift_out_kernel(
        const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> idata,
        torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> odata,
        const int top_to_target,
        const float bottom_weight,
        const int num_kernels
    ){
        const int batchsize = idata.size(0);
        const int channelsize = idata.size(1);
        const int freqsize = idata.size(2);
        const int timesize = idata.size(3);

        const int id = blockIdx.x * blockDim.x + threadIdx.x;
        const int time_id = id % timesize;
        const int freq_id = id / timesize;

        const int tf = freq_id - top_to_target;
        const int bf = freq_id - top_to_target - 1;   
        const float top_weight = 1 - bottom_weight;//0.2
        // shift = 1.8
        // 3    1.2  top
        // 2    0.2
        // 1 -> 0
        // 0    0    bottom

        if (id < num_kernels && bf >= 0 && tf < freqsize){ 
            for (int b = 0; b < batchsize; b++){
                for (int c = 0; c < channelsize; c++){                
                    odata[b][c][freq_id][time_id] = bottom_weight * idata[b][c][bf][time_id] + top_weight * idata[b][c][tf][time_id];
                }
            }
        }
    }

}// namespace

void interp_affine_out_cuda(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor indexes,
    torch::Tensor weights,
    int k,
    int n
){
    // input & output [batch,channel,freq,time]
    const int num_kernels = input.size(2)*input.size(3);
    
    // const int threads = std::min<int>(
        // at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, CUDA_MAX_THREADS);
    // I don't know why, but in my environment the above line malfunctions.
    const int threads = 1024;
    const dim3 blocks((num_kernels - 1)/threads + 1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "interp_affine_cuda", [&] {
    
            auto idata = input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits,size_t>();
            auto odata = output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits,size_t>();
            auto index_data = indexes.packed_accessor<int, 1, torch::RestrictPtrTraits,size_t>();
            auto weight_data = weights.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits,size_t>();
    
            interp_affine_out_kernel<scalar_t>
                <<<blocks, threads, 0, stream>>>(idata, odata, index_data, weight_data, num_kernels, k, n);
        }
    );
    AT_CUDA_CHECK(cudaGetLastError());
}

void interp_shift_out_cuda(
    torch::Tensor input,
    torch::Tensor output,
    float shift
){
    // input & output [batch,channel,freq,time]
    const int num_kernels = input.size(2) * input.size(3);
    const int threads = 1024;
    const dim3 blocks((num_kernels - 1) / threads + 1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int top_to_target = (int) std::floor(shift);//1
    const float bottom_weight = shift - (float) top_to_target;//0.8
    // shift = 1.8
    // 3    1.2
    // 2    0.2
    // 1 -> 0
    // 0    0
    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "interp_shift_cuda", [&] {
    
            auto idata = input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits,size_t>();
            auto odata = output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits,size_t>();
    
            interp_shift_out_kernel<scalar_t>
                <<<blocks, threads, 0, stream>>>(idata, odata, top_to_target, bottom_weight, num_kernels);
        }
    );
    AT_CUDA_CHECK(cudaGetLastError());
}
