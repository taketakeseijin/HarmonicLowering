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
        // const int nn = blockIdx.y;
        // const int step = blockIdx.x * blockDim.x + threadIdx.x;
        // const int freq_id = step * n + nn;
        const int freq_id = blockIdx.x * blockDim.x + threadIdx.x;
        const int step = freq_id / n;
        const int nn = freq_id % n;

        const int batchsize = idata.size(0);
        const int channels = idata.size(1);
        const int timesize = idata.size(2);
        const int freqsize = idata.size(3);

        const int ref_index_0 = step * k + index[nn];
        const int ref_index_1 = ref_index_0 + 1;
        const float weight_0 = weight[nn];
        const float weight_1 = 1 - weight_0;

        if (freq_id < num_kernels && ref_index_1 < freqsize){
            for (int b = 0; b < batchsize; b++){
                for (int c = 0; c < channels; c++){
                    for (int t = 0; t < timesize; t++){
                        odata[b][c][t][freq_id] = weight_0 * idata[b][c][t][ref_index_0] + weight_1 * idata[b][c][t][ref_index_1]; 
                    }
                }
            }
        }
    }

    template <typename scalar_t>
    __global__ void interp_shift_plus_out_kernel(
        const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> idata,
        torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> odata,
        const int top_to_target,
        const float bottom_weight,
        const int num_kernels
    ){

        // const int b = blockIdx.x;
        // const int c = blockIdx.y;
        // const int t = blockIdx.z;

        const int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < num_kernels){

            const int batchsize = idata.size(0);
            const int channelsize = idata.size(1);
            const int timesize = idata.size(2);
            const int freqsize = idata.size(3);

            const int b = id / (channelsize * timesize);
            const int c = (id / timesize) % channelsize;
            const int t = id % timesize;

            const float top_weight = 1 - bottom_weight;//0.2
            // shift = 1.8
            // 3    1.2  top
            // 2    0.2
            // 1 -> 0
            // 0    0    bottom
            float last_value = 0.0;
            for (int f = top_to_target; f < freqsize; f++){
                odata[b][c][t][f] = bottom_weight * last_value + top_weight * idata[b][c][t][f - top_to_target];
                last_value = idata[b][c][t][f - top_to_target];
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
    // input & output [batch,channel,time,freq]
    const int num_kernels = input.size(3);
    
    // const int threads = std::min<int>(
        // at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, CUDA_MAX_THREADS);
    // I don't know why, but in my environment the above line malfunctions.
    const int threads = 1024;
    const dim3 blocks(num_kernels/threads + 1);
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

void interp_shift_plus_out_cuda(
    torch::Tensor input,
    torch::Tensor output,
    float shift
){
    // input & output [batch,channel,time,freq]

    // const int threads = 1;
    // const dim3 blocks(input.size(0), input.size(1), input.size(2));

    const int num_kernels = input.size(0) * input.size(1) * input.size(2);
    const int threads = 1024;
    const dim3 blocks(num_kernels / threads + 1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int top_to_target = (int) std::floor(shift);//1
    const float bottom_weight = shift - (float) top_to_target;//0.8
    // shift = 1.8
    // 3    1.2
    // 2    0.2
    // 1 -> 0
    // 0    0
    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "interp_affine_cuda", [&] {
    
            auto idata = input.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits,size_t>();
            auto odata = output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits,size_t>();
    
            interp_shift_plus_out_kernel<scalar_t>
                <<<blocks, threads, 0, stream>>>(idata, odata, top_to_target, bottom_weight, num_kernels);
        }
    );
    AT_CUDA_CHECK(cudaGetLastError());
}
