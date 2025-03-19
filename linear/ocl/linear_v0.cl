__kernel void linear_kernel(__global const float* input,
                            __global const float* weight,
                            __global const float* bias,
                            __global float* output,
                            int batch_size,
                            int input_features,
                            int output_features) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < batch_size && col < output_features) {
        float sum = 0.0f;
        for (int i = 0; i < input_features; ++i) {
            sum += input[row * input_features + i] * 
                   weight[col * input_features + i];
        }
        sum += bias[col];
        output[row * output_features + col] = sum;
    }
}
__kernel void test_kernel(__global const float* input,
                    __global float* output,
                    unsigned int num) {
   int gid=get_global_id(0);
   if(gid<num){
        output[gid]=input[gid];
   }
}


// __global__ void linear_kernel(const float* input, const float* weight, const float* bias, float* output,
//                               int batch_size, int input_features, int output_features) {
//     int row = blockIdx.x * blockDim.x + threadIdx.x;
//     int col = blockIdx.y * blockDim.y + threadIdx.y;

//     if (row < batch_size && col < output_features) {
//         float sum = 0.0f;
//         for (int i = 0; i < input_features; ++i) {
//             sum += input[row * input_features + i] * weight[col * input_features + i];
//         }
//         sum += bias[col];
//         output[row * output_features + col] = sum;
//     }
// }