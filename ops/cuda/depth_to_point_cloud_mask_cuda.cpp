#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> depth_to_point_cloud_mask_cuda_forward(torch::Tensor depthmap);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> depth_to_point_cloud_mask_forward(torch::Tensor depthmap) {
    CHECK_INPUT(depthmap);

    return depth_to_point_cloud_mask_cuda_forward(depthmap);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &depth_to_point_cloud_mask_forward, "depth to point cloud mask forward");
}