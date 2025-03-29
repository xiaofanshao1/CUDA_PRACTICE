#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>


template<unsigned int BLOCK_SIZE, unsigned int STRIDE>
__global__ void cuda_sgemm(float* A_ptr, float* B_ptr, float* C_ptr, const int M, const int K, const int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    constexpr int SUPER_BLOCK_SIZE = BLOCK_SIZE * STRIDE;
    
    float* A_begin = A_ptr + blockIdx.y * SUPER_BLOCK_SIZE * K;
    float* B_begin = B_ptr + blockIdx.x * SUPER_BLOCK_SIZE;
    
    __shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];
    float sum[STRIDE][STRIDE] = {0.f};

    // 遍历所有需要的 K 维度块
    for(int s = 0; s < K; s +=BLOCK_SIZE) {
        // 对每个 STRIDE 区域计算
        for(int i = 0; i < STRIDE; i++) {
            for(int j = 0; j < STRIDE; j++) {
                // 加载当前块到共享内存
                if(s + tx < K && blockIdx.y * SUPER_BLOCK_SIZE + i * BLOCK_SIZE + ty < M) {
                    a_shared[ty][tx] = A_begin[(i * BLOCK_SIZE + ty) * K + s + tx];
                }
                
                if(s + ty < K && blockIdx.x * SUPER_BLOCK_SIZE + j * BLOCK_SIZE + tx < N) {
                    b_shared[ty][tx] = B_begin[s * N + ty * N + j * BLOCK_SIZE + tx];
                }
                
                __syncthreads();
                
                // 计算当前块的乘积
                for(int k = 0; k < BLOCK_SIZE && s + k < K; k++) {
                    sum[i][j] += a_shared[ty][k] * b_shared[k][tx];
                }
                
                __syncthreads();
            }
        }
    }

    // 写回结果
    for(int i = 0; i < STRIDE; i++) {
        for(int j = 0; j < STRIDE; j++) {
            int row = blockIdx.y * SUPER_BLOCK_SIZE + i * BLOCK_SIZE + ty;
            int col = blockIdx.x * SUPER_BLOCK_SIZE + j * BLOCK_SIZE + tx;
            if(row < M && col < N) {
                C_ptr[row * N + col] = sum[i][j];
            }
        }
    }
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
    constexpr int STRIDE=2;
    dim3 block(BLOCK_SIZE,BLOCK_SIZE);
    dim3 grid(divup(M,BLOCK_SIZE),divup(N,BLOCK_SIZE));
    
    cuda_sgemm<BLOCK_SIZE,STRIDE><<<grid,block>>>(matrix_A_d,matrix_B_d,matrix_C_d,M,K,N);
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





