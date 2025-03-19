import torch
from torch.utils.cpp_extension import load
'''
调用ninja编译成so文件进行链接,默认在cache文件当中
    /root/.cache/torch_extensions/py310_cu118
'''

linear_cuda = load(
    name="linear_cuda",  
    sources=["linear_v0_py.cu"], 
    extra_cflags=["-O2"],  # 额外编译选项
    verbose=True  # 显示编译日志
)

print("------JIT done-------")

batch_size = 32
in_features = 1024
out_features = 2048

input = torch.randn(batch_size, in_features).cuda()
weight = torch.randn(out_features, in_features).cuda()
bias = torch.randn(out_features).cuda()

# 调用自定义CUDA算子
output = linear_cuda.linear_forward(input, weight, bias)

# 验证结果
output_pytorch = torch.nn.functional.linear(input, weight, bias)
print("Max Error:", torch.max(torch.abs(output - output_pytorch)).item())