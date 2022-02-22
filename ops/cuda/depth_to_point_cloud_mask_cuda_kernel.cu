#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

#include <vector>

namespace {
// input: depthmap(b, h*w)
// output:  point_cloud (b, h*w, 3), mask (b, h*w)
__global__ void depth_to_point_cloud_mask_forward_kernel(
        const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> depthmap,
        torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> point_cloud,
        torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> mask,
        const int h, const int w){
    //batch index
    const int n = blockIdx.y;
    // column index
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c < depthmap.size(1)) {
        int d = depthmap[n][c];
        point_cloud[n][c][0] = c%w;
        point_cloud[n][c][1] = c/w;
        point_cloud[n][c][2] = d==0?1:d; // avoid dividing 0 in 3D to 2D transform
        mask[n][c] = (d!=0);
    }
}
} // namespace

// input: depthmap: (b, h, w, 1)
// output: point_cloud: (b, h*w, 3), mask: (b, h*w)
std::vector<torch::Tensor> depth_to_point_cloud_mask_cuda_forward(torch::Tensor depthmap) {
    const int b = depthmap.size(0);
    const int h = depthmap.size(1);
    const int w = depthmap.size(2);
    depthmap = depthmap.reshape({b, h*w});
    auto point_cloud = torch::zeros({b, h*w, 3},
        torch::TensorOptions().dtype(depthmap.scalar_type()).device(depthmap.device()));
    auto mask = torch::zeros({b, h*w}, torch::TensorOptions().dtype(torch::kInt32).device(depthmap.device()));

    const int threads = 1024;
    const dim3 blocks((h*w + threads - 1) / threads, b);

    AT_DISPATCH_INTEGRAL_TYPES(depthmap.scalar_type(), "depth_to_point_cloud_mask_forward_cuda", ([&]() {
        depth_to_point_cloud_mask_forward_kernel<<<blocks, threads>>>(
                depthmap.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                point_cloud.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
                mask.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                h, w);
        }));
    getLastCudaError("depth_to_point_cloud_mask_forward_kernel() execution failed.");
    checkCudaErrors(cudaDeviceSynchronize());
    return {point_cloud, mask};
}

