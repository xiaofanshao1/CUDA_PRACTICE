
#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32
/**
 * 
 */
__global__ void reduce2(float* d_intput,float * d_output){
    float *input_begin_global=d_intput+blockDim.x * blockIdx.x *2;
    __shared__ float input_begin[THREAD_PER_BLOCK];
    input_begin[threadIdx.x]=input_begin_global[threadIdx.x]+ input_begin_global[threadIdx.x+THREAD_PER_BLOCK];
    
    __syncthreads();

    //完全展开for循环，可以通过编译器的宏
    {
        #pragma unroll
        for(int interval=blockDim.x/2;interval>0;interval/=2){
            if(threadIdx.x<interval){
                input_begin[threadIdx.x]+=input_begin[threadIdx.x+interval];
            }
            if(interval>WARP_SIZE)
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
        printf("%f ",a[i]);
    }
    printf("\n");
}



int main(){
    const int N=32 * 1024 * 1024;
    float* h_input= (float*)malloc(N*sizeof(float));
    float* d_intput;
    cudaMalloc((void**)&d_intput,N*sizeof(float));
    
    constexpr int block_num=(N+THREAD_PER_BLOCK-1)/THREAD_PER_BLOCK/2;
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
            for(int j=0;j<2 * THREAD_PER_BLOCK;j++){
                tsum+=h_input[ 2 * THREAD_PER_BLOCK*i+j];
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