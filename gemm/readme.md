# SGEMM

#### gemm/sgemm_v0_global_mem.cu

- block[16,16]，每个block映射到C的block上。block内`thread[i,j]`需要load来自于A和B的两个向量数据 read:`2K` write:`1` 

<img src="./assets/image-20250321132845930.png" alt="image-20250321132845930" style="zoom: 33%;" />

