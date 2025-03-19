#include <torch/extension.h>

__global__ void linear_kernel(const float* input, const float* weight, const float* bias, float* output,
                              int batch_size, int input_features, int output_features) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < batch_size && col < output_features) {
        float sum = 0.0f;
        for (int i = 0; i < input_features; ++i) {
            sum += input[row * input_features + i] * weight[col * input_features + i];
        }
        sum += bias[col];
        output[row * output_features + col] = sum;
    }
}
torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int batch_size = input.size(0);
    int input_features = input.size(1);
    int output_features = weight.size(0);

    auto output = torch::zeros({batch_size, output_features}, input.options());

    dim3 threads(16, 16);
    dim3 blocks((batch_size + threads.x - 1) / threads.x, (output_features + threads.y - 1) / threads.y);

    linear_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_features,
        output_features
    );

    return output;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_forward", &linear_forward, "Linear forward CUDA");
}