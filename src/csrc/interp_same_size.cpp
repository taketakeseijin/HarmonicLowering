#include <torch/extension.h>
#include <vector>

// CUDA declarations
void interp_affine_out_cuda(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor indexes,
    torch::Tensor weights,
    int k,
    int n,
    bool time_last
    );

void interp_shift_out_cuda(
    torch::Tensor input,
    torch::Tensor output,
    float shift,
    bool time_last
    );

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void interp_affine_out(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor indexes,
    torch::Tensor weights,
    int k,
    int n,
    bool time_last
    ) {
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  CHECK_INPUT(indexes);
  CHECK_INPUT(weights);
  output.zero_();

  interp_affine_out_cuda(input, output, indexes, weights, k, n, time_last);
}

void interp_shift_out(
    torch::Tensor input,
    torch::Tensor output,
    float shift,
    bool time_last
    ) {
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  // AT_ASSERTM(shift > 0, "shift must be plus number");
  output.zero_();
  interp_shift_out_cuda(input, output, shift, time_last);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("interp_affine", &interp_affine_out, "affine interpolation with same size (CUDA)");
  m.def("interp_shift", &interp_shift_out, "shift interpolation with same size (CUDA)");
}