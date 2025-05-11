
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
// M=2048 N=512
// thread&data block[32,8] ->  grid[64,64]


__global__ void transpose_naive_v0(float* input, float* output,const int MATRIX_M,const int MATRIX_N )
{

    //transpose符合矩阵分块规律 
    
    //1. 思路一：整体一起考虑，成为一个大的block（因为没有引入sharem mem，可以跨越原来分的block划分，作为一个大整体来进行操作
    //2. 思路二：分块+块内操作
    //for block wise:  swap row&col
    //for block internal: swap row&col

    //下面采用大整体操作，合并所有block，在大整体上进行操作
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_idx < MATRIX_N && row_idx < MATRIX_M) {

       int idx = row_idx * MATRIX_N + col_idx;
       int trans_idx = col_idx * MATRIX_M + row_idx;
       output[trans_idx] = input[idx];
    }
}
void transpose_cpu(float *input,float *output, const int M,const int N){
    for(int m=0;m<M;m++){ 
        for(int n=0;n<N;n++){
            const int input_index=m*N+n;
            const int output_index=n*M+m;
            output[output_index]=input[input_index];
        }
    }
}

class Perf
{
public:
    Perf(const std::string& name) {
        m_name = name;
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_end);
        cudaEventRecord(m_start);
        cudaEventSynchronize(m_start);
    }

    ~Perf() {
        cudaEventRecord(m_end);
        cudaEventSynchronize(m_end);
        float elapsed_time = 0.0;
        cudaEventElapsedTime(&elapsed_time, m_start, m_end);
        std::cout << m_name << " elapse: " << elapsed_time << " ms" << std::endl;
    }

private:
    std::string m_name;
    cudaEvent_t m_start, m_end;
}; // class Perf


bool check(float *cpu_result,float* gpu_result, const int M, const int N){
    const int size =M*N;
    for(int i=0;i<size;i++){
        if(cpu_result[i]!=gpu_result[i])
            return false;
    }
    return true;
}

int main(){
    const int MATRIX_M=2048;
    const int MATRIX_N=512;
    const size_t size= MATRIX_M *MATRIX_N;

    float *input_host=(float *)malloc(size*sizeof(float));
    float *output_host_cpu_calc=(float*)malloc(size*sizeof(float));
    float *output_host_gpu_calc=(float*)malloc(size*sizeof(float));

    for(int i=0;i<size;i++){
        input_host[i]= 2.0 *(float)drand48()-1.0;
    }

    transpose_cpu(input_host,output_host_cpu_calc,MATRIX_M,MATRIX_N);
    float *input_device,*output_device;

    cudaMalloc(&input_device, size*sizeof(float));
    cudaMemcpy(input_device,input_host,size*sizeof(float),cudaMemcpyHostToDevice);
    cudaMalloc(&output_device,size*sizeof(float));

    //=============
    cudaMemset(output_device,0,size*sizeof(float));

    for(int i=0;i<5;i++){
        Perf perf("transepose_native_32_8");
        dim3 block_size(32,8);
        dim3 grid_size((MATRIX_N-1)/block_size.x +1,(MATRIX_M-1)/block_size.y+1);
        transpose_naive_v0<<<grid_size,block_size>>>(input_device,output_device,MATRIX_M,MATRIX_N);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(output_host_gpu_calc,output_device,
                size*sizeof(float),cudaMemcpyDeviceToHost);
    if(check(output_host_cpu_calc,output_host_gpu_calc, MATRIX_M,MATRIX_N)){
        std::cout<<"right"<<std::endl;
    }else{
        std::cout<<"wrong"<<std::endl; 
    }

}