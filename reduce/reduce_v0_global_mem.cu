#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>


__global__ void reduce0(float* d_intput,float * d_output){

    float *input_begin=d_intput+blockDim.x * blockIdx.x;
    for(int idx=1;idx<blockDim.x;idx=idx*2){
        if(threadIdx.x%(idx*2) == 0)
        //这个地方+idx写成了1,错了很难检查出来
            input_begin[threadIdx.x]+= input_begin[threadIdx.x+idx];
        __syncthreads();
    }
    
    if(threadIdx.x==0)
        d_output[blockIdx.x]=input_begin[0];

}
bool check(float* a,float*b,int n){
 /**
 * @brief Effective Modern C++ 对于浮点数有特点说明
 * - 一般精度范围为 [1e-5, 1e-9]
 * - 一般应用 1e-5 即可
 */
    for(int i=0;i<n;i++){
        if(abs(a[i]-b[i])>0.005) return false;
    }
    return true;
}
void printArray(float* a,int n){
    for(int i=0;i<10;i++){
        printf("%f ",a[i]);
    }
    printf("\n");
}

#define THREAD_PER_BLOCK 256

int main(){
    const int N=32 * 1024 * 1024;
    float* h_input= (float*)malloc(N*sizeof(float));
/**
 * @brief 在设备（GPU）上分配内存
 * 
 * 在 C/C++ 中，如果函数需要修改指针的值，必须传递指针的地址（即双重指针）。
 * 如果直接传递 d_input，cudaMalloc 只能修改 d_input 的副本，而不会影响外部的 d_input。
 */
    float* d_intput;
    cudaMalloc((void**)&d_intput,N*sizeof(float));
    
    //const expr和const有什么区别？ 一个在编译期间求值，另一个在运行期
    constexpr int block_num=(N+THREAD_PER_BLOCK-1)/THREAD_PER_BLOCK;
    float *h_output= (float*)malloc(block_num*sizeof(float));
   
    float *d_output;
    cudaMalloc((void**)&d_output,block_num*sizeof(float));

    float* result=(float*) malloc(block_num*sizeof(float));
    
    // initialize the input
    {
/**
 * @quesiton 这个rand是哪里来的? 
 * 
 * @details
 * 生成一个均匀分布的伪随机数,drand48 生成一个均匀分布的伪随机数，范围在 [0.0, 1.0) 之间。
 * 在代码中，`2.0 * drand48() - 1.0` 将随机数的范围扩展到 [-1.0, 1.0)。
 * 
 * drand48 是 POSIX 标准库的一部分，定义在 `<stdlib.h>` 头文件中。
 * 它是基于线性同余算法实现的伪随机数生成器。
 * 
 * @note
 * 在现代 C++ 中，推荐使用 `<random>` 库中的随机数生成器，如 `std::uniform_real_distribution`。
 * 
 * @return double 返回一个在 [0.0, 1.0) 范围内的伪随机数
 */
        for(int i=0;i<N;i++){
            h_input[i]=2.0*drand48()-1.0;
        }
    }

    // compute on cpu side
    {
        for(int i=0;i<block_num;i++){
            float tsum=0;
            for(int j=0;j<THREAD_PER_BLOCK;j++){
                tsum+=h_input[THREAD_PER_BLOCK*i+j];
            }
            h_output[i]=tsum;
        }
    }
    // compute on GPU
    {
        cudaMemcpy(d_intput,h_input,N*sizeof(float),cudaMemcpyHostToDevice);
        //这个应该是大写还是小写也可以,这个不这样写怎么填xy?
        //可以在<<<grid_dim,block_dim>>>直接做，默认是一维
        dim3 Grid(block_num,1);
        dim3 Block(THREAD_PER_BLOCK,1);
        
        reduce0<<<Grid,Block>>>(d_intput,d_output);
    }
    //check & release
    {
        
        cudaMemcpy(result,d_output,block_num*sizeof(float),cudaMemcpyDeviceToHost);

        if(check(result,h_output,block_num)){
            printf("ans is good\n");
        }else{
            printf("ans is wrong\n");
            printArray(result,block_num);
            printArray(h_output,block_num);
            
        }

        cudaFree(d_intput);
        cudaFree(d_output);
        
        free(h_input);
        free(result);
        free(h_output);
    }

    return 0;
    
}