#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>


#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*> (&(pointer))[0])

template<unsigned int M_NUM_PER_BLOCK, unsigned int K_NUM_PER_BLOCK, unsigned int N_NUM_PER_BLOCK,
    unsigned int M_NUM_PER_THREAD,
    unsigned int N_NUM_PER_THREAD,
    unsigned int K_NUM_PER_THREAD>
__global__ void cuda_sgemm(float* A_ptr, float* B_ptr, float* C_ptr, const int M, const int K, const int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float *A_begin=A_ptr+ blockIdx.y *M_NUM_PER_BLOCK*K;
    float *B_begin=B_ptr+ blockIdx.x*N_NUM_PER_BLOCK;

    //A发生了旋转
    __shared__ float a_shared[K_NUM_PER_BLOCK][M_NUM_PER_BLOCK];
    __shared__ float b_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];

    float a_reg[M_NUM_PER_THREAD]{};
    float b_reg[N_NUM_PER_THREAD]{};
    float a_load_reg[K_NUM_PER_THREAD]{};
    float sum[M_NUM_PER_THREAD][N_NUM_PER_THREAD]{};
    
    for(int s=0;s<K;s+=K_NUM_PER_BLOCK){
        for(int i=0;i<M_NUM_PER_THREAD;i++){
            FETCH_FLOAT4(a_load_reg[0]) 
                =FETCH_FLOAT4(A_begin[K*(ty*M_NUM_PER_THREAD+i)+tx*K_NUM_PER_THREAD+s]);
            a_shared[tx*K_NUM_PER_THREAD][ty*M_NUM_PER_THREAD+i]=a_load_reg[0];
            a_shared[tx*K_NUM_PER_THREAD+1][ty*M_NUM_PER_THREAD+i]=a_load_reg[1];
            a_shared[tx*K_NUM_PER_THREAD+2][ty*M_NUM_PER_THREAD+i]=a_load_reg[2];
            a_shared[tx*K_NUM_PER_THREAD+3][ty*M_NUM_PER_THREAD+i]=a_load_reg[3];
            // FETCH_FLOAT4(a_shared[ty*M_NUM_PER_THREAD+i][tx*K_NUM_PER_THREAD])=
            //     FETCH_FLOAT4(A_begin[K*(ty*M_NUM_PER_THREAD+i)+tx*K_NUM_PER_THREAD+s]);
        }
        for(int i=0;i<K_NUM_PER_THREAD;i++){
            FETCH_FLOAT4(b_shared[ty*K_NUM_PER_THREAD+i][tx*N_NUM_PER_THREAD])=
                FETCH_FLOAT4(B_begin[N*(ty*K_NUM_PER_THREAD+s+i)+tx*N_NUM_PER_THREAD]);
        }
        __syncthreads();
       
        for(int k=0;k<K_NUM_PER_BLOCK;k++){

            // a_reg[0]=a_shared[ty * M_NUM_PER_THREAD][k];
            // a_reg[1]=a_shared[ty * M_NUM_PER_THREAD+1][k];
            // a_reg[2]=a_shared[ty * M_NUM_PER_THREAD+2][k];
            // a_reg[3]=a_shared[ty * M_NUM_PER_THREAD+3][k];
            FETCH_FLOAT4(a_reg[0])= FETCH_FLOAT4(a_shared[k][ty*K_NUM_PER_THREAD] );
            FETCH_FLOAT4(b_reg[0])=FETCH_FLOAT4(b_shared[k][tx*N_NUM_PER_THREAD]) ;

            for(int i=0;i<M_NUM_PER_THREAD;i++){
                for(int j=0;j<N_NUM_PER_THREAD;j++){
                    sum[i][j]+=a_reg[i]*b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    float *C_begin=C_ptr+ N * blockIdx.y *M_NUM_PER_BLOCK+blockIdx.x*N_NUM_PER_BLOCK;

    for(int i=0;i<M_NUM_PER_THREAD;i++){
        for(int j=0;j<N_NUM_PER_THREAD;j++){
            C_begin[N*(ty*M_NUM_PER_THREAD+i)+tx*N_NUM_PER_THREAD+j]=sum[i][j];
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
    //重新划分线程dispatch
    //block内线程[16,16] 
    //1. 每个线程处理[4,4] 16的tile,float4方便处理[4,4]，外积也方便
    constexpr int M_NUM_PER_BLOCK=64;
    constexpr int N_NUM_PER_BLOCK=64;
    constexpr int K_NUM_PER_BLOCK=64;
    constexpr int  NUM_PER_THREAD=16;
    constexpr int  M_NUM_PER_THREAD=4;
    constexpr int  N_NUM_PER_THREAD=4;
    constexpr int  K_NUM_PER_THREAD=4;
    dim3 block(16,16);
    dim3 grid(M/M_NUM_PER_BLOCK,N/N_NUM_PER_BLOCK);
    
    cuda_sgemm<M_NUM_PER_BLOCK,K_NUM_PER_BLOCK,N_NUM_PER_BLOCK,M_NUM_PER_THREAD,N_NUM_PER_THREAD,K_NUM_PER_THREAD><<<grid,block>>>(matrix_A_d,matrix_B_d,matrix_C_d,M,K,N);
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





