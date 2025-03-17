
#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32

/**
 * 思路：用warp内shuffle进行reduce
 *  shuffle down 向左平移，相当于<<
 */ 
template<unsigned int NUM_PER_BLOCK>
__global__ void reduce2(float* d_intput,float * d_output){
    float *input_begin_global=d_intput+ NUM_PER_BLOCK * blockIdx.x;

    //此处不再使用映射每个thread号的shared mem，而是直接reg进行累加 

    float sum=0.;
    for(int idx=0;idx<NUM_PER_BLOCK/THREAD_PER_BLOCK;idx++){
        sum+=input_begin_global[threadIdx.x+idx*THREAD_PER_BLOCK];
    }
    /**
     * 1. 不同warp当中都reduce到一个float当中
     */
    for(int warp_interval=WARP_SIZE/2;warp_interval>0;warp_interval/=2){
        sum+=__shfl_down_sync(0xffffffff,sum,warp_interval);
    }
    /**
     * 2. 用share mem来收集之前所有warp0，reduce到一个warp可以处理的数量。注意这时候收集的是一个block内所有的warp
     */
    //是否可以用{}初始化呢
    __shared__ float warp_sum[WARP_SIZE];
    
    //这个地方之前忘记初始化,
    {
        if (threadIdx.x < WARP_SIZE) {
                warp_sum[threadIdx.x] = 0.0f;
        }
        __syncthreads();
    }
   

    const int laneID=threadIdx.x%WARP_SIZE;
    const int warpID=threadIdx.x/WARP_SIZE;
    if(laneID==0){
        warp_sum[warpID%32]=sum;//注意使用赋值而不是累加，这个地方之前造成了错误
    }
    //这个地方之前漏掉了,这个地方各个block reduce到share mem之后需要一次sync
    __syncthreads();
    
    
    /**
     * 3. block内用第一个warp进行求和 share mem
     */
    if(warpID==0){
        //复用sum变量
        sum=0.;
        sum=warp_sum[laneID];//之前填了为thread.x
        for(int interval=WARP_SIZE/2;interval>0;interval/=2){
            sum+=__shfl_down_sync(0xffffffff,sum,interval);
        }
        if(laneID==0) d_output[blockIdx.x]=sum;//之前填了为thread.x
    }

}
bool check(float* a,float*b,int n){
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



int main(){
    const int N=32 * 1024 * 1024;
    float* h_input= (float*)malloc(N*sizeof(float));
    float* d_intput;
    cudaMalloc((void**)&d_intput,N*sizeof(float));
    
    constexpr int block_num=1024;
    constexpr int num_per_block=N/block_num;
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
        //TODO:
        for(int i=0;i<block_num;i++){
            float tsum=0;
            for(int j=0;j<num_per_block;j++){
                tsum+=h_input[num_per_block*i+j];
            }
            h_output[i]=tsum;
        }
    }
    // compute on GPU
    {
        cudaMemcpy(d_intput,h_input,N*sizeof(float),cudaMemcpyHostToDevice);
        dim3 Grid(block_num,1);
        dim3 Block(THREAD_PER_BLOCK,1);
        
        reduce2<num_per_block><<<Grid,Block>>>(d_intput,d_output);
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