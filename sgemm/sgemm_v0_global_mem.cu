#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

#define divup(a, b) (((a) + (b) - 1) / (b))

#define A(i,j) matrix[(i)*n+(j)]
void random_init(int m,int n,float *matrix){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            A(i,j)=2.0 *(float)drand48()-1.0;
        }
    }
}

void sgemm_cpu(float *A_ptr,float *B_ptr,float* C_ptr,
    const int m,const int k,const int n){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            float tmp=0.f;
            for(int kk=0;kk<k;kk++){
                tmp+=A_ptr[k*i+kk]*B_ptr[n*kk+j];
            }
            C_ptr[i*n+j]=tmp;
        }
    }
}
__global__ void cuda_sgemm(float* A_ptr,float* B_ptr, float* C_ptr,const int M,const int K,const int N){
    float* A_begin= A_ptr + blockIdx.y * blockDim.y * K;
    float* B_begin= B_ptr + blockIdx.x * blockDim.x;

    float sum=0.f;
    //1. Block [16,16] 256个线程，x和y两个线程号bundle在一起代表唯一线程ID 
    //2. C block[16,16]与 thread block[16,16]一一对应，[x,y]号线程完成C[x,y]
    //3. [x,y]线程计算来自A和B两个向量的内积   dot( [1,K] ,[K,1] )
    for(int k=0;k < K; k++){
        sum+= A_begin[threadIdx.y*K+k] * B_begin[k*N+threadIdx.x];
    }

    const int x=threadIdx.x+blockDim.x*blockIdx.x;
    const int y=threadIdx.y+blockDim.y*blockIdx.y;
    C_ptr[y*N+x]=sum;
}

float checkMatrix(float* a,float* b,const int m, const int n){
    //a from cpu, b from gpu
    bool bHit{};
    float max_diff=0.f;
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            max_diff=max(max_diff,abs(a[n*i+j]-b[n*i+j]));
            if((max_diff>0.1f) && !bHit) {
                printf("first hit error: i %d j %d diff %f got %f expect %f\n",i,j,max_diff,b[n*i+j],a[n*i+j]);
                bHit=true;
            }
        }
    }
    if(!bHit) printf("result is good!\n");
    return max_diff;
}

//Q1: cuda.h 和 cuda_runtime.h 暴露api的简单介绍
/** 
*   1. cuda.h (Driver API) 上下文细粒度控制 	-lcuda
    2. cuda_runtime.h  -lcuda
*/
int main(){
    constexpr int M=2048;
    constexpr int N=2048;
    constexpr int K=2048;

    //Q2: size_t的历史是怎么回来的?
    //  可以追溯到C的历史，表达平台上最大位数的无符号数 和指针保持一致
    constexpr size_t mem_size_A=M*K*sizeof(float);
    constexpr size_t mem_size_B=K*N*sizeof(float);
    constexpr size_t mem_size_C=M*N*sizeof(float);

    float* maxtrix_A_host=(float*)malloc(mem_size_A);
    float* maxtrix_B_host=(float*)malloc(mem_size_B);
    float* matrix_C_host_calc=(float*)malloc(mem_size_C);
    float* matrix_C_device_calc=(float*)malloc(mem_size_C);
     
    float *matrix_C_d; 
    float *matrix_A_d; 
    float *matrix_B_d; 
    cudaMalloc((void **)&matrix_C_d, mem_size_C);
    cudaMalloc((void **)&matrix_A_d, mem_size_A);
    cudaMalloc((void **)&matrix_B_d, mem_size_B);


    //init matrix
    //Q3:memset从什么库里面引入?
    // <cstring>与<string>不同，在c标准库当中提供的是 内存操作和字符数组处理函数. char当做byte使用
    {
        random_init(M,K,maxtrix_A_host);
        random_init(K,N,maxtrix_B_host);
        memset(matrix_C_host_calc,0,mem_size_C);
        memset(matrix_C_device_calc,0,mem_size_C);

        //注意这个matrix不再是**
        cudaMemcpy(matrix_A_d,maxtrix_A_host,mem_size_A,cudaMemcpyHostToDevice);
        cudaMemcpy(matrix_B_d,maxtrix_B_host,mem_size_B,cudaMemcpyHostToDevice);
    //Q4:对于C这个矩阵一般是怎么初始化的，也是memset?

    }

    constexpr int BLOCK=16;
    dim3 block(BLOCK,BLOCK);
    dim3 grid(divup(M,BLOCK),divup(N,BLOCK));
    
    cuda_sgemm<<<grid,block>>>(matrix_A_d,matrix_B_d,matrix_C_d,M,K,N);
    sgemm_cpu(maxtrix_A_host,maxtrix_B_host,matrix_C_host_calc,M,K,N);


    // check the result
    {
        cudaMemcpy(matrix_C_device_calc,matrix_C_d,mem_size_C,cudaMemcpyDeviceToHost);
        checkMatrix(matrix_C_host_calc,matrix_C_device_calc,M,N);
    }
    //clean resource
    {
        free(maxtrix_A_host);
        free(maxtrix_B_host);
        free(matrix_C_host_calc);
        free(matrix_C_device_calc);
        cudaFree(matrix_A_d);
        cudaFree(matrix_B_d);
        cudaFree(matrix_C_d);

    }


    return 0;
}




