
#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256

//问题 targetlink里面都是什么库？
__global__ void reduce2(float* d_intput,float * d_output){
    __shared__ float input_begin[THREAD_PER_BLOCK];

    float *input_begin_global=d_intput+blockDim.x * blockIdx.x;
    input_begin[threadIdx.x]=input_begin_global[threadIdx.x];
    __syncthreads();

    //1. 在v当中之前每个warp都是一半执行另一半不执行。这一版当中做出改进，让一个warp尽量做同样的事情
    //2. 在前128个线程也就是前4个warp都参与计算，后面4个warp不参与。原来交叉线程不做事，现在改变线程做事分布
    //3. 这是一种非常别扭的做法，线程号和数据号关系难找
    {
    for(int sg=1;sg<blockDim.x;sg=sg*2){
        //每组内只一个thread实际工作，计算出需要的thread数目
        int thread_num=blockDim.x/(sg*2);
        //安排让前一半的线程工作
        if(threadIdx.x<thread_num){
            int group_start=threadIdx.x *2*sg;
            input_begin[group_start]+= input_begin[group_start+sg];
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