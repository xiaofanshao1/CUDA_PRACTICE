
//M=2048， N=512
//threads block[16,64]  -> ??


#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
// M=2048 N=512
// thread&data block[32,8] ->  grid[64,64]


__global__ void transpose_float4(float* input, float* output,const int MATRIX_M,const int MATRIX_N)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    // 这里每个线程处理4行，4列，因此需要除以4
    if (col_idx >= (MATRIX_N >> 2) or row_idx >= (MATRIX_M >> 2)) {
        return;
    }

    // 取4*4矩阵的技巧: block thread和data 已经映射情况好下，一个线程处理4*4，可以直接如下取左上角
    int offset = (row_idx * MATRIX_N + col_idx) << 2; // * 4
    float4* input_v4 = reinterpret_cast<float4*>(input + offset);

    // 以向量的方式拿到4 * 4矩阵， 第n行地址偏移 = (MATRIX_N / 4) * n
    float4 src_row0 = input_v4[0];
    float4 src_row1 = input_v4[MATRIX_N >> 2];
    float4 src_row2 = input_v4[MATRIX_N >> 1];
    float4 src_row3 = input_v4[(MATRIX_N >> 2) * 3];

    // 线程内在寄存器当中进行专职 4 * 4小矩阵转置
    float4 dst_row0 = make_float4(src_row0.x, src_row1.x, src_row2.x, src_row3.x);
    float4 dst_row1 = make_float4(src_row0.y, src_row1.y, src_row2.y, src_row3.y);
    float4 dst_row2 = make_float4(src_row0.z, src_row1.z, src_row2.z, src_row3.z);
    float4 dst_row3 = make_float4(src_row0.w, src_row1.w, src_row2.w, src_row3.w);

    // 将4行写入
    offset = (col_idx * MATRIX_M + row_idx) << 2;
    float4* dst_v4 = reinterpret_cast<float4*>(output + offset);
    dst_v4[0] = dst_row0;
    dst_v4[MATRIX_M >> 2] = dst_row1;
    dst_v4[MATRIX_M >> 1] = dst_row2;
    dst_v4[(MATRIX_M >> 2) * 3] = dst_row3;
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
        transpose_float4<<<grid_size,block_size>>>(input_device,output_device,MATRIX_M,MATRIX_N);
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