# Reduce pattern 

reduce/reduce_v0_global_mem.cu

在global mem上以block单位进行reduce，注意block内分组技巧

```cpp
  float *input_begin=d_intput+blockDim.x * blockIdx.x;
  for(int sg=1;sg<blockDim.x;sg=sg*2){
      if(threadIdx.x%(sg*2) == 0)
          input_begin[threadIdx.x]+= input_begin[threadIdx.x+sg];
      __syncthreads();
  }
```



---



reduce/reduce_v1_shared_mem.cu

每个线程各自的数据从global mem 加载到 shared mem中

```cpp
  __shared__ float input_begin[THREAD_PER_BLOCK];
  float *input_begin_global=d_intput+blockDim.x * blockIdx.x;
  input_begin[threadIdx.x]=input_begin_global[threadIdx.x];
  __syncthreads();
```



---



reduce/reduce_v2_warp_divergence.cu


---

reduce/reduce_v3_no_bank_conflict.cu

Reduce合并分组方式改变，线程都load同一个bank数据

```
  for(int interval=blockDim.x/2;interval>0;interval/=2){
      if(threadIdx.x<interval){
          input_begin[threadIdx.x]+=input_begin[threadIdx.x+interval];
      }
      __syncthreads();
  }
```





---



reduce/reduce_v4_block_utilization.cu

---



reduce/reduce_v5_unroll_last_warp.cu

```cpp
for(int interval=blockDim.x/2;interval>0;interval/=2){
    if(threadIdx.x<interval){
        input_begin[threadIdx.x]+=input_begin[threadIdx.x+interval];
    }
    //添加判断是否reduce的group大于一个warp，这个建立在我们预知一个warp一起完成
    if(interval>WARP_SIZE)
        __syncthreads();
}
```



---



reduce/reduce_v6_unroll_all_warp.cu

```cpp
#pragma unroll
for(int interval=blockDim.x/2;interval>0;interval/=2){
    if(threadIdx.x<interval){
        input_begin[threadIdx.x]+=input_begin[threadIdx.x+interval];
    }
    if(interval>WARP_SIZE)
        __syncthreads();
}
```

---



reduce/reduce_v7_multi_add.cu
reduce/reduce_v8_warp_shuffle.cu









### 
