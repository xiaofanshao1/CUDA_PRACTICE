#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << ": "   \
                  << cudaGetErrorString(status) << std::endl;                  \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        std::cerr << "CUBLAS Error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << status << std::endl;                                      \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

int main() {
    // 矩阵维度 (M=512, N=512, K=1024)
    const int M = 512, N = 512, K = 1024;

    // 分配主机内存 (A[M,K], B[K,N], C[M,N])
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    // 初始化数据 (示例: A全1, B全2)
    std::fill(h_A, h_A + M * K, 1.0f);
    std::fill(h_B, h_B + K * N, 2.0f);

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    // 拷贝数据到设备
    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // 创建 cuBLAS 句柄
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // 执行 GEMM: C = alpha * A * B + beta * C
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  // 不转置A和B
        M, N, K,                    // 矩阵维度
        &alpha,                     // alpha
        d_A, CUDA_R_32F, M,         // A矩阵 (行主序)
        d_B, CUDA_R_32F, K,         // B矩阵 (行主序)
        &beta,                      // beta
        d_C, CUDA_R_32F, M,         // C矩阵 (行主序)
        CUDA_R_32F,                 // 计算精度
        CUBLAS_GEMM_DEFAULT         // 算法选择
    ));

    // 拷贝结果回主机
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // 验证结果 (C[0,0] 应为 1*2*1024=2048)
    std::cout << "C[0] = " << h_C[0] << " (预期: 2048)" << std::endl;

    // 释放资源
    delete[] h_A; delete[] h_B; delete[] h_C;
    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}
