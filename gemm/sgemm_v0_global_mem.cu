#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

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

}

bool checkMatrix(float* a,float* b,const int m, const int n){
    return true;
}

//cuda.h 和 cuda_runtime.h 暴露api的简单介绍
int main(){
    constexpr int M=2048;
    constexpr int N=2048;
    constexpr int K=2048;

    //size_t的历史是怎么回来的
    //可以追溯到C的历史，表达平台上最大位数的无符号数 和指针保持一致
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
    //memset从什么库里面引入
    {
        random_init(M,K,maxtrix_A_host);
        random_init(K,N,maxtrix_B_host);
        memset(matrix_C_host_calc,0,mem_size_C);
        memset(matrix_C_device_calc,0,mem_size_C);

        //注意这个matrix不再是**
        cudaMemcpy(matrix_A_d,maxtrix_A_host,mem_size_A,cudaMemcpyHostToDevice);
        cudaMemcpy(matrix_B_d,maxtrix_B_host,mem_size_B,cudaMemcpyHostToDevice);
        //对于C这个矩阵一般是怎么初始化的，也是memset?

    }

    constexpr int BLOCK=16;
    dim3 block(BLOCK,BLOCK);
    dim3 grid((M+BLOCK-1)/BLOCK,(N+BLOCK-1)/BLOCK);
    
    cuda_sgemm<<<grid,block>>>(matrix_A_d,matrix_B_d,matrix_C_d,M,K,N);


    // check the result
    {

    }

    return 0;
}




