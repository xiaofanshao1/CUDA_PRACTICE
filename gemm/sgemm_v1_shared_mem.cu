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
    
    //两个shared mem 向量块
    __shared__ float a_shared[BLOCK_SIZE][_K];
    __shared__ float b_shared[_K][BLOCK_SIZE];
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
    /**
     * 1. 当使用2048时编译失败
     * a_shared大小 = 16 × 2048 × 4 = 131,072 字节 (128KB)
     * 也就是一个BLOCK会用掉总和 = 128KB + 128KB = 256KB
     * 比如使用-arch=compute_52（Maxwell架构） 编译器就会使用48KB共享内存限制
     * 
     * 2. 修改为compute86 下1024，kernel会调用128KB,但是仍然似乎编译器限制在48KB？
     * 通过添加set(CMAKE_CUDA_ARCHITECTURES "86")，会增加到8.6 共享内存限制到128KB
     * cmake会添加/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler   -g --generate-code=arch=compute_86,code=[compute_86,sm_86]
     *  仍然有限制[build] ptxas error   : Entry function '_Z10cuda_sgemmILj16ELj1024EEvPfS0_S0_iii' uses too much shared data (0x20000 bytes, 0xc000 max)
     *  -  0x20000 = 131,072字节（128KB） → 实际需求
     *  -  0xc000 = 49,152字节（48KB） → 编译器使用的限制
     */
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
