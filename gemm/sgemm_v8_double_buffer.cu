#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>


#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*> (&(pointer))[0])


/**
 * Double buffer to hide:
 */
template<unsigned int BLOCK_SIZE_M,//height of tile each thread block in charge with size C 
         unsigned int BLOCK_SIZE_N, 
         unsigned int BLOCK_SIZE_K,
        unsigned int THREAD_SIZE_Y,//height of tile each thread in charge
        unsigned int THREAD_SIZE_X>
__global__ void cuda_sgemm(float* A_ptr, float* B_ptr, float* C_ptr, const int M, const int K, const int N) {
    //Tile index 
    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //thread id in a block,double buffer
    const int tid= ty*blockDim.x+tx;
    __shared__ float a_shared[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float b_shared[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

    // focus on sliding window on Global mem 
    float *A_begin=A_ptr+by * BLOCK_SIZE_M *K;
    float *B_begin=B_ptr+bx * BLOCK_SIZE_N;

    float sum[THREAD_SIZE_Y][THREAD_SIZE_X]{};
    float reg_a[THREAD_SIZE_Y]{};//处理A，对应SplitM
    float reg_b[THREAD_SIZE_X]{};//处理B，对应SplitN
    float load4_rega[4]{};

    //dispatch逻辑   
    // [BLOCK_SIZE_M,BLOCK_SIZE_N]->[128,128]
    // each thread [THREAD_SIZE_X][THREAD_SIZE_Y]-> [8,8]  thread block[16,16]

    // load data 此时进行线程重排编号保证一行同时用float4来 load
    //例如 对于B，BLOCK_SIZE[128,8] 横向需要32（每个线程内有一个float4这是横向一起load的原因）,这时候把[16,16]->[32,8] 对于竖向可以每个线程load一行，
    //例如 对于A，[16,16]->[2,128]
    const int A_LOAD_BLOCK_THREAD_ROW= BLOCK_SIZE_K/4;//比如此时BLOCK_SIZE_K=8，横向两个线程即可一次通过两个float4发送8个float
    const int B_LOAD_BLOCK_THREAD_ROW= BLOCK_SIZE_N/4;

    const int A_LOAD_tid_x=tid%A_LOAD_BLOCK_THREAD_ROW;
    const int A_LOAD_tid_y=tid/A_LOAD_BLOCK_THREAD_ROW;
    const int B_LOAD_tid_x=tid%B_LOAD_BLOCK_THREAD_ROW;
    const int B_LOAD_tid_y=tid/B_LOAD_BLOCK_THREAD_ROW;
    
    // load to shared 0
    {
        //load to a_shared from global
        FETCH_FLOAT4(load4_rega[0])=FETCH_FLOAT4(A_begin[A_LOAD_tid_y*K+A_LOAD_tid_x*4]);//一行的8 float存在两个横向线程的float4当中
        a_shared[0][A_LOAD_tid_x*4][A_LOAD_tid_y]=load4_rega[0];
        a_shared[0][A_LOAD_tid_x*4+1][A_LOAD_tid_y]=load4_rega[1];
        a_shared[0][A_LOAD_tid_x*4+2][A_LOAD_tid_y]=load4_rega[2];
        a_shared[0][A_LOAD_tid_x*4+3][A_LOAD_tid_y]=load4_rega[3];

        //load to b_shared from global,无需借助寄存器来进行中转
        FETCH_FLOAT4(b_shared[0][B_LOAD_tid_y][B_LOAD_tid_x*4])=FETCH_FLOAT4(B_begin[N*B_LOAD_tid_y + B_LOAD_tid_x*4]);
        __syncthreads();

    }

    
    int write_stage=1;
    for(int s=BLOCK_SIZE_K;s<K;s+=BLOCK_SIZE_K){
        //SplitK 对K纬度划分成BLOCK_SIZE_K的块
        FETCH_FLOAT4(load4_rega[0])=FETCH_FLOAT4(A_begin[A_LOAD_tid_y*K+A_LOAD_tid_x*4+s]);
        a_shared[write_stage][A_LOAD_tid_x*4][A_LOAD_tid_y]=load4_rega[0];
        a_shared[write_stage][A_LOAD_tid_x*4+1][A_LOAD_tid_y]=load4_rega[1];
        a_shared[write_stage][A_LOAD_tid_x*4+2][A_LOAD_tid_y]=load4_rega[2];
        a_shared[write_stage][A_LOAD_tid_x*4+3][A_LOAD_tid_y]=load4_rega[3];
        FETCH_FLOAT4(b_shared[write_stage][B_LOAD_tid_y][B_LOAD_tid_x*4])=FETCH_FLOAT4(B_begin[N*(B_LOAD_tid_y+s) + B_LOAD_tid_x*4]);
        
         __syncthreads();
        write_stage=write_stage^1;
        for(int k=0;k<BLOCK_SIZE_K;k++){
            //对BLOCK_SIZE_K进行外积累加
            FETCH_FLOAT4(reg_a[0])= FETCH_FLOAT4(a_shared[write_stage][k][ty*THREAD_SIZE_Y] );
            FETCH_FLOAT4(reg_a[4])= FETCH_FLOAT4(a_shared[write_stage][k][ty*THREAD_SIZE_Y+4] );
            FETCH_FLOAT4(reg_b[0])=FETCH_FLOAT4(b_shared[write_stage][k][tx*THREAD_SIZE_X]) ;
            FETCH_FLOAT4(reg_b[4])=FETCH_FLOAT4(b_shared[write_stage][k][tx*THREAD_SIZE_X+4]) ;

            for(int i=0;i<THREAD_SIZE_Y;i++){
                for(int j=0;j<THREAD_SIZE_X;j++){
                    sum[i][j]+=reg_a[i]*reg_b[j];
                }
            }
            __syncthreads();
        }
   }
    //last step
   {
        write_stage =write_stage^1;
        for(int k=0;k<BLOCK_SIZE_K;k++){
            //对BLOCK_SIZE_K进行外积累加
            FETCH_FLOAT4(reg_a[0])= FETCH_FLOAT4(a_shared[write_stage][k][ty*THREAD_SIZE_Y] );
            FETCH_FLOAT4(reg_a[4])= FETCH_FLOAT4(a_shared[write_stage][k][ty*THREAD_SIZE_Y+4] );
            FETCH_FLOAT4(reg_b[0])=FETCH_FLOAT4(b_shared[write_stage][k][tx*THREAD_SIZE_X]) ;
            FETCH_FLOAT4(reg_b[4])=FETCH_FLOAT4(b_shared[write_stage][k][tx*THREAD_SIZE_X+4]) ;

            for(int i=0;i<THREAD_SIZE_Y;i++){
                for(int j=0;j<THREAD_SIZE_X;j++){
                    sum[i][j]+=reg_a[i]*reg_b[j];
                }
            }
        }
   }
 
    float *C_begin=C_ptr+ N * by *BLOCK_SIZE_M+bx*BLOCK_SIZE_N;

    for(int i=0;i<THREAD_SIZE_Y;i++){ 
        //对于
        FETCH_FLOAT4(C_begin[N*(ty*THREAD_SIZE_Y+i)+tx*THREAD_SIZE_X])=FETCH_FLOAT4(sum[i][0]);
        FETCH_FLOAT4(C_begin[N*(ty*THREAD_SIZE_Y+i)+tx*THREAD_SIZE_X+4])=FETCH_FLOAT4(sum[i][4]);
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

    constexpr int BLOCK_SIZE_M=128;
    constexpr int BLOCK_SIZE_K=8;
    constexpr int BLOCK_SIZE_N=128;
    constexpr int THREAD_SIZE_Y=8;
    constexpr int THREAD_SIZE_X=8;

    dim3 block(BLOCK_SIZE_N/THREAD_SIZE_X,BLOCK_SIZE_M/THREAD_SIZE_Y);
    dim3 grid(N/BLOCK_SIZE_N,M/BLOCK_SIZE_M);
    
    cuda_sgemm<BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        THREAD_SIZE_Y,
        THREAD_SIZE_X><<<grid,block>>>(matrix_A_d,matrix_B_d,matrix_C_d,M,K,N);
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





