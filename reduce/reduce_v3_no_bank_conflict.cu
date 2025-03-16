
#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256

__global__ void reduce2(float* d_intput,float * d_output){
    __shared__ float input_begin[THREAD_PER_BLOCK];

    float *input_begin_global=d_intput+blockDim.x * blockIdx.x;
    input_begin[threadIdx.x]=input_begin_global[threadIdx.x];
    __syncthreads();
    /**
     * 一个block内同一个warp不同thread访问了同一个share mem bank发生冲突,其会产生先后同步。主要检查第一个warp
     * 在v1版本当中：
     * 线程号和数据号映射比较直接
     *  thread0 -> data[0 1]
     *  thread31 -> data[31 32]
     * data 0 32都在0号bank当中引发2路冲突 
     * 
     * 在v2版本当中
     * thread0 -> data[0 1]
     * thread16 -> data[32 33]
     * data 0 32引发2路冲突
     * 
     * 在v3版本当中
     *  对于thread.x读取和写入都是同一个bank，不发生conflict
     *  thread0 -> data[0 128] 
     *  thread8 -> data[8 136]
     *  thread16 -> data[16 144]
     *  thread24 -> data[24 152]
     *  ---
     *  这里的技巧主要是映射，线程号 线程内读取数据时候，所有数据号映射到32bank空间当中是同一的。  
     *  比如  0 128都是在0号bank上
     */
    //对比v1,当中之前每个warp都是一半执行另一半不执行。这一版每一个warp也是尽量做同样的事情
    {
        for(int interval=blockDim.x/2;interval>0;interval/=2){
            if(threadIdx.x<interval){
                input_begin[threadIdx.x]+=input_begin[threadIdx.x+interval];
            }
            __syncthreads();
        }

    }
    
    if(threadIdx.x==0)
        d_output[blockIdx.x]=input_begin[0];

}
bool check(float* a,float*b,int n){
    for(int i=0;i<n;i++){
        if(abs(a[i]-b[i])>0.005) return false;
    }
    return true;
}
void printArray(float* a,int n){
    for(int i=0;i<10;i++){
// 1. 当我引入cstdio的时候，使用printf不需要引入命名空间吗  std::printf()??
// 是编译器行为，cpp标准里面应该加上std::才是对的
        
        printf("%f ",a[i]);
    }
    printf("\n");
}



int main(){
    const int N=32 * 1024 * 1024;
    float* h_input= (float*)malloc(N*sizeof(float));
    float* d_intput;
    cudaMalloc((void**)&d_intput,N*sizeof(float));
    
    constexpr int block_num=(N+THREAD_PER_BLOCK-1)/THREAD_PER_BLOCK;
    float *h_output= (float*)malloc(block_num*sizeof(float));
   
    float *d_output;
    cudaMalloc((void**)&d_output,block_num*sizeof(float));

    float* result=(float*) malloc(block_num*sizeof(float));
    

    {
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
        dim3 Grid(block_num,1);
        dim3 Block(THREAD_PER_BLOCK,1);
        
        reduce2<<<Grid,Block>>>(d_intput,d_output);
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