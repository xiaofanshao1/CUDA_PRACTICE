#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*> (&(pointer))[0])

template<unsigned int M_NUM_PER_BLOCK, unsigned int K_NUM_PER_BLOCK, unsigned int N_NUM_PER_BLOCK,unsigned int NUM_PER_THREAD>
__global__ void cuda_sgemm(float* A_ptr, float* B_ptr, float* C_ptr, const int M, const int K, const int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float *A_begin=A_ptr+ blockIdx.y *M_NUM_PER_BLOCK*K;
    float *B_begin=B_ptr+ blockIdx.x*N_NUM_PER_BLOCK;

    __shared__ float a_shared[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float b_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];
    constexpr int REG_NUM=NUM_PER_THREAD/2;

    /**
     *  利用register tile来做外积
     *  1. 线程如何分工 [8,32]
     *    按照这个假设仍然是需要每个线程写出4个数，可以安排 rm=rn=2  结果是4个数
     *    这里为了方便，在load完数据之后进行线程的索引重排 变成[16,16]
     */
    float sum[REG_NUM][REG_NUM]{};
    float a_reg[REG_NUM]{};
    float b_reg[REG_NUM]{};
    int tid=ty*blockDim.x+tx;
    int ctx= tid%16; int cty=tid/16;
    
    for(int s=0;s<K;s+=K_NUM_PER_BLOCK){
        FETCH_FLOAT4(a_shared[ty][tx * NUM_PER_THREAD])=FETCH_FLOAT4(A_begin[K*ty +s +tx* NUM_PER_THREAD]);
        FETCH_FLOAT4(b_shared[ty][tx* NUM_PER_THREAD])=FETCH_FLOAT4(B_begin[N*(ty+s)+tx*NUM_PER_THREAD]);
        __syncthreads();
       
        for(int k=0;k<K_NUM_PER_BLOCK;k++){
            a_reg[0]=a_shared[cty *2][k];
            a_reg[1]=a_shared[cty*2+1][k];
            b_reg[0]=b_shared[k][ctx*2];
            b_reg[1]=b_shared[k][ctx*2+1];
            for(int i=0;i<REG_NUM;i++){
                for(int j=0;j<REG_NUM;j++){
                    sum[i][j]+=a_reg[i]*b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    float *C_begin=C_ptr+ N * blockIdx.y *M_NUM_PER_BLOCK+blockIdx.x*N_NUM_PER_BLOCK;
    //这个地方也换成二维的循环
    for(int i=0;i<REG_NUM;i++){
        for(int j=0;j<REG_NUM;j++){
            C_begin[N*(cty*2+i)+ctx*2+j]=sum[i][j];
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

    constexpr int M_NUM_PER_BLOCK=32;
    constexpr int N_NUM_PER_BLOCK=32;
    constexpr int K_NUM_PER_BLOCK=32;
    constexpr int  NUM_PER_THREAD=4;
    dim3 block(8,32);
    dim3 grid(M/M_NUM_PER_BLOCK,N/N_NUM_PER_BLOCK);
    
    cuda_sgemm<M_NUM_PER_BLOCK,K_NUM_PER_BLOCK,N_NUM_PER_BLOCK,NUM_PER_THREAD><<<grid,block>>>(matrix_A_d,matrix_B_d,matrix_C_d,M,K,N);
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





