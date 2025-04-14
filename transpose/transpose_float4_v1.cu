
//M=2048， N=512
//threads block[16,64]  -> ??
__global__ void transpose_float4(float* input, float* output)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    // 这里每个线程处理4行，4列，因此需要除以4
    if (col_idx >= (MATRIX_N >> 2) or row_idx >= (MATRIX_M >> 2)) {
        return;
    }

    // 取4*4矩阵的技巧: block thread和data 已经映射情况好下，一个线程处理4*4，可以直接如下取左上角
    int offset = (row_idx * MATRIX_N + col_idx) << 2; // * 4
    float4* input_v4 = reinterpret_cast<float4*>(input + offset);

    // 以向量的方式拿到4 * 4矩阵， 第n行地址偏移 = (MATRIX_N / 4) * n
    float4 src_row0 = input_v4[0];
    float4 src_row1 = input_v4[MATRIX_N >> 2];
    float4 src_row2 = input_v4[MATRIX_N >> 1];
    float4 src_row3 = input_v4[(MATRIX_N >> 2) * 3];

    // 线程内在寄存器当中进行专职 4 * 4小矩阵转置
    float4 dst_row0 = make_float4(src_row0.x, src_row1.x, src_row2.x, src_row3.x);
    float4 dst_row1 = make_float4(src_row0.y, src_row1.y, src_row2.y, src_row3.y);
    float4 dst_row2 = make_float4(src_row0.z, src_row1.z, src_row2.z, src_row3.z);
    float4 dst_row3 = make_float4(src_row0.w, src_row1.w, src_row2.w, src_row3.w);

    // 将4行写入
    offset = (col_idx * MATRIX_M + row_idx) << 2;
    float4* dst_v4 = reinterpret_cast<float4*>(output + offset);
    dst_v4[0] = dst_row0;
    dst_v4[MATRIX_M >> 2] = dst_row1;
    dst_v4[MATRIX_M >> 1] = dst_row2;
    dst_v4[(MATRIX_M >> 2) * 3] = dst_row3;
}