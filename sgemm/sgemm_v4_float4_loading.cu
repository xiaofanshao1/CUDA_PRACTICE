#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

/**
 * block(16,16)->block(8,32) 对于横向8个线程一次4个float，可以一次拿满
 */

// Q1: 这个为什么这样写呢？
// 1. 将一个float* 指针指向连续的4个float内存指针 float4*
//      例如float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};float4 vec = FETCH_FLOAT4(data[0]); 
// 2. 步骤解释如下

/* 分步解析（假设调用 FETCH_FLOAT4(some_float)）：
 * 
 * 1. &(pointer) 
 *    - 取 pointer 的地址（pointer可以是单个float或float数组中的元素）
 *    - 例如：若 pointer = float_array[0]，则 &pointer 得到 &float_array[0]（类型：float*）
 * 
 * 2. reinterpret_cast<float4*>(...)
 *    - 强制类型转换：将 float* 指针转换为 float4* 指针
 *    - 关键点：不改变内存实际数据，仅改变编译器对这段内存的解释方式
 *    - 例如：float* → float4* 表示"将接下来的16字节（4*float）当作一个float4结构体"
 * 
 * 3. [0]
 *    - 对转换后的 float4* 指针取第一个元素
 *    - 由于 float4* 是指向 float4 的指针，[0] 即解引用为 float4 对象
 *    - 效果：从 pointer 的地址开始，连续读取4个float到float4的x/y/z/w成员
 * 
 * 示例：
 *   float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
 *   float4 vec = FETCH_FLOAT4(data[0]); 
 *   // vec.x = 1.0f, vec.y = 2.0f, vec.z = 3.0f, vec.w = 4.0f
 * 
 * 注意事项：
 *   - 内存对齐：pointer 的地址必须是16字节对齐（即地址 % 16 == 0）
 *   - 安全边界：pointer 后必须有至少3个连续float，否则越界
 */

// ==== reinterpret_cast vs static_cast 核心区别 ====
// 1. reinterpret_cast:
//    - 用途: 内存二进制位的完全重新解释（无关类型强制转换）
//    - 特点: 高风险、无类型检查、编译期直接操作内存
//    - 场景: 指针类型强制转换（如 float* -> float4*）
// 2. static_cast:
//    - 用途: 相关类型间的安全转换（需编译器已知转换规则）
//    - 特点: 相对安全、有类型检查、可能插入运行时逻辑
//    - 场景: 数值类型转换、基类/派生类转换

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*> (&(pointer))[0])

template<unsigned int M_NUM_PER_BLOCK, unsigned int K_NUM_PER_BLOCK, unsigned int N_NUM_PER_BLOCK,unsigned int NUM_PER_THREAD>
__global__ void cuda_sgemm(float* A_ptr, float* B_ptr, float* C_ptr, const int M, const int K, const int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float *A_begin=A_ptr+ blockIdx.y *M_NUM_PER_BLOCK*K;
    float *B_begin=B_ptr+ blockIdx.x*N_NUM_PER_BLOCK;

    __shared__ float a_shared[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float b_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];
    float sum[NUM_PER_THREAD]{0.f};
    
    for(int s=0;s<K;s+=K_NUM_PER_BLOCK){
        FETCH_FLOAT4(a_shared[ty][tx * NUM_PER_THREAD])=FETCH_FLOAT4(A_begin[K*ty +s +tx* NUM_PER_THREAD]);
        // a_shared[ty][tx * NUM_PER_THREAD]=A_begin[K*ty +s +tx* NUM_PER_THREAD];
        // a_shared[ty][tx * NUM_PER_THREAD+1]=A_begin[K*ty +s +tx* NUM_PER_THREAD+1];
        // a_shared[ty][tx * NUM_PER_THREAD+2]=A_begin[K*ty +s +tx* NUM_PER_THREAD+2];
        // a_shared[ty][tx * NUM_PER_THREAD+3]=A_begin[K*ty +s +tx* NUM_PER_THREAD+3];
        FETCH_FLOAT4(b_shared[ty][tx* NUM_PER_THREAD])=FETCH_FLOAT4(B_begin[N*(ty+s)+tx*NUM_PER_THREAD]);
        // b_shared[ty][tx* NUM_PER_THREAD]=B_begin[N*(ty+s)+tx*NUM_PER_THREAD];
        __syncthreads();
       
        //compute 
        for(int i=0;i<NUM_PER_THREAD;i++){
            for(int k=0;k<K_NUM_PER_BLOCK;k++){
                // [8 ,32]
                sum[i]+=a_shared[ty][k] * b_shared[k][tx*NUM_PER_THREAD+i];
            }
        }
        __syncthreads();
    }

    float *C_begin=C_ptr+ N * blockIdx.y *M_NUM_PER_BLOCK+blockIdx.x*N_NUM_PER_BLOCK;
    for(int i=0;i<NUM_PER_THREAD;i++){
        //仍然是一个线程负责4列，也就是负责填回原来的4个pixel
        C_begin[ty*N +tx*NUM_PER_THREAD+i]=sum[i];
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





