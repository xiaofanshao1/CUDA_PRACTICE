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

template<unsigned int BLOCK_SIZE,unsigned int _K>
__global__ void cuda_sgemm(float* A_ptr,float* B_ptr, float* C_ptr,const int M,const int K,const int N){
    float* A_begin= A_ptr + blockIdx.y * blockDim.y * K;
    float* B_begin= B_ptr + blockIdx.x * blockDim.x;
    const int x=threadIdx.x+blockDim.x*blockIdx.x;
    const int y=threadIdx.y+blockDim.y*blockIdx.y;
    
    //两个shared mem 向量块
    __shared__ float a_shared=[BLOCK_SIZE][_K];
    __shared__ float b_shared=[_K][BLOCK_SIZE];
    //避免加载重复:
    //1. 可以让主对角线上的x=y的线程做加载
    //2. 按照tile进行加载
    
    //此处因为blockDim.y==blockDim.x，所以s可以 both 对thread.x 和 thread.y进行index
    for(int s=0;s < K ;s+= blockDim.x){
        a_shared[threadIdx.y][threadIdx.x+s]=A_begin[threadIdx.x+s + threadIdx.y * K];
        b_shared[threadIdx.y+s][threadIdx.x]=B_begin[threadIdx.x + (s + threadIdx.y)*N];
    }
    __syncthreads();

    float sum=0.f;
    for(int k=0;k < K; k++){
        sum+= a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
    }

    C_ptr[y*N+x]=sum;
}

bool checkMatrix(float* a,float* b,const int m, const int n){
    return true;
}

int main(){
    //这个地方需要变小一点，对shared mem压力比较大
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


    //init matrix
    {
        random_init(M,K,maxtrix_A_host);
        random_init(K,N,maxtrix_B_host);
        memset(matrix_C_host_calc,0,mem_size_C);
        memset(matrix_C_device_calc,0,mem_size_C);

        cudaMemcpy(matrix_A_d,maxtrix_A_host,mem_size_A,cudaMemcpyHostToDevice);
        cudaMemcpy(matrix_B_d,maxtrix_B_host,mem_size_B,cudaMemcpyHostToDevice);

    }

    constexpr int BLOCK=16;
    dim3 block(BLOCK,BLOCK);
    dim3 grid((M+BLOCK-1)/BLOCK,(N+BLOCK-1)/BLOCK);
    
    cuda_sgemm<<<grid,block>>>(matrix_A_d,matrix_B_d,matrix_C_d,M,K,N);


    {

    }

    return 0;
}




