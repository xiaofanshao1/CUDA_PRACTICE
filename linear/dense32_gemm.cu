#include <torch/extension.h>
#define divup(a, b) (((a) + (b) - 1) / (b))
//问题：
//1. batch的大小在中间变化怎么处理，可以切batch的方式？最后一个batch匹配其他的kernel？
__global__ void gemm_native(
    const float* __restrict__ input,    /*[batch_size,input_features]  */
    const float* __restrict__ weight,   /*[input_features,output_features]*/
    const float* __restrict__ bias,     /*[batch_size,output_features]*/
    float* __restrict__ output,
    int batch_size,
    int input_features,
    int output_features
) {
    //直接使用global mem来做


 
}

__global__ void gemm_32x32_shared_splitK(
    const float* __restrict__ input,    /*[batch_size,input_features]  */
    const float* __restrict__ weight,   /*[input_features,output_features]*/
    const float* __restrict__ bias,     /*[batch_size,output_features]*/
    float* __restrict__ output,
    int batch_size,
    int input_features,
    int output_features
) {

    /**
     * 1.  
     * 2. fuse bias add
     */
    // 共享内存声明（静态分块）
    __shared__ float X_shared[32][32];  // 输入分块缓存 (4KB)
    __shared__ float W_shared[32][32];  // 权重分块缓存 (4KB)

    
 
}


torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {

// //1. 问题定义：y= Wx+b 也就是最基础的大矩阵GEMM  [32,2048]
// 输入：X 的形状为 [batch_size=32, 1024]
// 权重：W 的形状为 [1024, 2048]
// 输出：Y = X @ W + b，形状 [32, 2048]
// //2. 分块逻辑 以[32,1024]@[1024,2048]为例  
// grid.x = ceil(2048 / 32) = 64（覆盖输出列）
// grid.y = ceil(32 / 32) = 1（覆盖batch）

/**
 * 优化矩阵乘法核函数（分块策略：32x32输出分块）
 * 
 * 分块配置（对应表格第1行）：
 *   - 输出分块大小: 32x32       // 每个Block计算一个32x32的输出子矩阵
 *   - Block线程布局: 32x8       // 256线程/Block 让x方向满足32达成warp。此时y纬度每个处理4行 
 *   - Grid布局: (64,1,1)        // 覆盖2048列: ceil(2048/32)=64
 *   - 共享内存用量: 8KB          // X分块32x32 + W分块32x32 = 8KB shared mem
 */

    int batch_size = input.size(0);
    int input_features = input.size(1);
    int output_features = weight.size(0);

    auto output = torch::zeros({batch_size, output_features}, input.options());

    constexpr int BLOCK_SIZE=32;
    dim3 threads(32, 8);  // 每个Block 32x8线程
    dim3 blocks(
        divup(output_features,BLOCK_SIZE),  // x方向分块数
        divup(batch_size,BLOCK_SIZE)        // y方向分块数
    );


    linear_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_features,
        output_features
    );

    return output;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_forward", &linear_forward, "Linear forward CUDA");
}