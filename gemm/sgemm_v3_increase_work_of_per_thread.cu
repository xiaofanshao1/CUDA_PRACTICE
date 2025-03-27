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

template<unsigned int BLOCK_SIZE,unsigned int STRIDE>
__global__ void cuda_sgemm(float* A_ptr,float* B_ptr, float* C_ptr,const int M,const int K,const int N){
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    constexpr int SUPER_BLOCK_SIZE = BLOCK_SIZE *STRIDE;
    float* A_begin= A_ptr + blockIdx.y * SUPER_BLOCK_SIZE * K;
    float* B_begin= B_ptr + blockIdx.x * SUPER_BLOCK_SIZE;
    //shared mem仍然覆盖each block 处理的所有数据点
    __shared__ float a_shared=[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_shared=[BLOCK_SIZE][BLOCK_SIZE];
    float sum[STRIDE][STRIDE]{0.f};

    //tile load & compute
    //block thread每次 处理一个 STRIDE *BLOCK_SIZE块 (对于MK矩阵当中)
    for(int s=0;s < K ;s+= SUPER_BLOCK_SIZE){
        for(int i=0;i<STRIDE;i++){
            for(int j=0;j<STRIDE;j++){
                //用i j进行滑动的时候，注意对A和B的pattern  
                // s[1+/*注释*/ +1]这样可以过编译器吗
                //顺序按照 滑块 -> Y -> X 来做
                a_shared[ty][tx]=A_begin[s+      ty*K+i*BLOCK_SIZE*K   + tx+j*BLOCK_SIZE];
                b_shared[ty][tx]=B_begin[s*N+    ty*N+ j*BLOCK_SIZE*N  + tx+ i*BLOCK_SIZE];

                for(int k=0;k<block;k++){
                    sum[i][j]+=A_shared[ty][k] * B_shared[k][tx];
                }
            }
        }   
    }

    for(int i=0;i<STRIDE;i++){
        for(int j=0;j<STRIDE;j++){
            C_ptr[] =sum[i][j];
        }
    }
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
    constexpr int STRIDE=2;
    dim3 block(BLOCK,BLOCK);
    dim3 grid((M+BLOCK-1)/BLOCK/STRIDE,(N+BLOCK-1)/BLOCK/STRIDE);
    
    cuda_sgemm<BLOCK,STRIDE><<<grid,block>>>(matrix_A_d,matrix_B_d,matrix_C_d,M,K,N);

    {
    }

    return 0;
}




