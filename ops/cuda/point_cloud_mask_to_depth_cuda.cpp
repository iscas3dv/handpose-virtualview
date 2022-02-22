#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor point_cloud_mask_to_depth_cuda_forward(torch::Tensor point_cloud, torch::Tensor mask,
        const int h, const int w);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor point_cloud_mask_to_depth_forward(torch::Tensor point_cloud, torch::Tensor mask,
        const int h, const int w) {
    CHECK_INPUT(point_cloud);

    return point_cloud_mask_to_depth_cuda_forward(point_cloud, mask, h, w);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &point_cloud_mask_to_depth_forward, "point cloud mask to depth forward");
}