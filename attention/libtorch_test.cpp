#include <torch/torch.h>
#include <iostream>

int main(){
    // 在CPU上生成随机张量
    torch::Tensor a_cpu = torch::rand({2, 3});
    torch::Tensor b_cpu = torch::rand({2, 3});
    torch::Tensor c_cpu = a_cpu + b_cpu;

    // 将张量复制到GPU
    torch::Tensor a_gpu = a_cpu.to("cuda");  
    torch::Tensor b_gpu = b_cpu.to("cuda");  
    torch::Tensor c_gpu = a_gpu + b_gpu;

    torch::Tensor c_gpu_cpu = c_gpu.to("cpu");  

   

    std::cout << "CPU result:\n" << c_cpu << std::endl;
    std::cout << "\nGPU result (on CPU):\n" << c_gpu_cpu << std::endl;
    std::cout << "\nDelta" <<c_gpu_cpu-c_cpu  << std::endl;

    return 0;

}