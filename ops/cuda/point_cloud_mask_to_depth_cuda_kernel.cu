#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

#include <vector>

namespace {
// input:  point_cloud (b,h*w,3)
// output: depthmap(b,h*w), mask (b, h*w)
__global__ void point_cloud_mask_to_depth_forward_kernel(
        torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> depthmap,
        const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> point_cloud,
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> mask,
        const int h, const int w) {
    //batch index
    const int n = blockIdx.y;
    // column index
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c < depthmap.size(1) && mask[n][c]) {
        int u = point_cloud[n][c][0], v = point_cloud[n][c][1], d = point_cloud[n][c][2];
        if(0<=u && u<w && 0<=v && v<h) {
            atomicMin(&depthmap[n][v*w+u], d);
        }
    }
}

// input: depthmap(b, h*w)
// output: depthmap(b, h*w)
__global__ void set_background_forward_kernel(
        torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> depthmap,
        const int h, const int w, const int bg_val) {
    //batch index
    const int n = blockIdx.y;
    // column index
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c < depthmap.size(1) && depthmap[n][c] == INT_MAX) {
        depthmap[n][c] = bg_val;
    }
}
} // namespace

// input: point_cloud: (b, h*w, 3), mask: (b, h*w)
// output: depthmap: (b, h, w, 1)
torch::Tensor point_cloud_mask_to_depth_cuda_forward(torch::Tensor point_cloud, torch::Tensor mask,
        const int h, const int w) {
    const int b = point_cloud.size(0);
    auto depthmap = torch::full({b, h*w}, INT_MAX,
        torch::TensorOptions().dtype(point_cloud.scalar_type()).device(point_cloud.device()));

    const int threads = 1024;
    const dim3 blocks((h*w + threads - 1) / threads, b);

    AT_DISPATCH_INTEGRAL_TYPES(depthmap.scalar_type(), "point_cloud_mask_to_depth_forward_cuda", ([&]() {
        point_cloud_mask_to_depth_forward_kernel<<<blocks, threads>>>(
                depthmap.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                point_cloud.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
                mask.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                h, w);
    }));
    getLastCudaError("point_cloud_mask_to_depth_forward_kernel() execution failed.");
    checkCudaErrors(cudaDeviceSynchronize());

    int bg_val = 0;
    AT_DISPATCH_INTEGRAL_TYPES(depthmap.scalar_type(), "set_background_forward_kernel", ([&]() {
        set_background_forward_kernel<<<blocks, threads>>>(
                depthmap.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                h, w, bg_val);
    }));
    getLastCudaError("set_background_forward_kernel() execution failed.");
    checkCudaErrors(cudaDeviceSynchronize());
    depthmap = depthmap.reshape({b, h, w, 1});
    return depthmap;
}