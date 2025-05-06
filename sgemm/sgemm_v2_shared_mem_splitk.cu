#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>


template<unsigned int BLOCK_SIZE,unsigned int _K>
__global__ void cuda_sgemm(float* A_ptr,float* B_ptr, float* C_ptr,const int M,const int K,const int N){
    float* A_begin= A_ptr + blockIdx.y * blockDim.y * K;
    float* B_begin= B_ptr + blockIdx.x * blockDim.x;
    const int x=threadIdx.x+blockDim.x*blockIdx.x;
    const int y=threadIdx.y+blockDim.y*blockIdx.y;
    
    //两个shared mem 用来 sliding tile from A&B, 用C tile累加结果 
    /**
     *     Q1. 问题 CUDA是否支持cpp11有花括号初始化？
     *     不支持,一般用线程的方式使用时进行初始化。shared mem在编译时无法确定地址，初始化可能引发race condition等
     */

    __shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];

    //tile load & compute
    float sum=0.f;
    // here assum blockDim.x==blockDim.y, Split K and sliding
    for(int s=0;s < K ;s+= blockDim.x){
        //load data to shared mem
        a_shared[threadIdx.y][threadIdx.x]=A_begin[threadIdx.y * K + s + threadIdx.x];
        b_shared[threadIdx.y][threadIdx.x]=B_begin[s*N+threadIdx.y*N+threadIdx.x];//注意这里面不需要单独+s，因为是只复制过来竖条块
        __syncthreads();
        //这个地方注意转换
        for(int k=0;k<BLOCK_SIZE;k++){
            sum+=a_shared[threadIdx.y][k]*b_shared[k][threadIdx.x]; 
        }
        __syncthreads();
        
    }
    //滑动窗口完成
    C_ptr[y*N+x] =sum;
}


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

int main(){
    constexpr int M=128;
    constexpr int N=128;
    constexpr int K=128;

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

    {
        random_init(M,K,maxtrix_A_host);
        random_init(K,N,maxtrix_B_host);
        memset(matrix_C_host_calc,0,mem_size_C);
        memset(matrix_C_device_calc,0,mem_size_C);

        cudaMemcpy(matrix_A_d,maxtrix_A_host,mem_size_A,cudaMemcpyHostToDevice);
        cudaMemcpy(matrix_B_d,maxtrix_B_host,mem_size_B,cudaMemcpyHostToDevice);

    }

    constexpr int BLOCK_SIZE=16;
    dim3 block(BLOCK_SIZE,BLOCK_SIZE);
    dim3 grid(divup(M,BLOCK_SIZE),divup(N,BLOCK_SIZE));
    
    cuda_sgemm<BLOCK_SIZE,K><<<grid,block>>>(matrix_A_d,matrix_B_d,matrix_C_d,M,K,N);
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




