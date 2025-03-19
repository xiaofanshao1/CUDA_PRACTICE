#include <iostream>
#include <vector>
#include <fstream>
#include <CL/cl.h>

// 错误检查宏
#define CHECK_CL_ERROR(err, msg) \
    if (err != CL_SUCCESS) { \
        std::cerr << "OpenCL Error (" << err << "): " << msg << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; \
        throw std::runtime_error(msg); \
    }

class OpenCLEnv {
public:
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    OpenCLEnv() {
        // 1. 获取平台
        cl_int err = clGetPlatformIDs(1, &platform, nullptr);
        CHECK_CL_ERROR(err, "Failed to get platforms");

        // 2. 获取设备
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        CHECK_CL_ERROR(err, "Failed to get device");

        // 3. 创建上下文
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        CHECK_CL_ERROR(err, "Failed to create context");

        // 4. 创建命令队列
        queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
        CHECK_CL_ERROR(err, "Failed to create command queue");
    }

    ~OpenCLEnv() {
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

class OpenCLBuffer {
    cl_mem buffer;
    cl_context context;
public:
    OpenCLBuffer(cl_context ctx, size_t size, void* host_ptr, cl_mem_flags flags) : context(ctx) {
        cl_int err;
        buffer = clCreateBuffer(context, flags, size, host_ptr, &err);
        CHECK_CL_ERROR(err, "Buffer creation failed");
    }

    ~OpenCLBuffer() { if (buffer) clReleaseMemObject(buffer); }
    operator cl_mem() const { return buffer; }
};

int main() try {

    // 配置参数
    const int batch_size = 64;
    const int input_features = 1024;
    const int output_features = 512;

    // 初始化数据 (使用 C++ 容器)
    std::vector<float> input(batch_size * input_features, 1.0f);
    std::vector<float> weight(output_features * input_features, 0.5f);
    std::vector<float> bias(output_features, 0.1f);
    std::vector<float> output(batch_size * output_features, 0.0f);

    // 初始化 OpenCL 环境 (RAII 管理)
    OpenCLEnv env;

    // 创建缓冲区 (RAII 管理)
    OpenCLBuffer input_buf(env.context, input.size() * sizeof(float), input.data(), 
                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    OpenCLBuffer weight_buf(env.context, weight.size() * sizeof(float), weight.data(), 
                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    OpenCLBuffer bias_buf(env.context, bias.size() * sizeof(float), bias.data(), 
                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    OpenCLBuffer output_buf(env.context, output.size() * sizeof(float), nullptr, 
                          CL_MEM_WRITE_ONLY);

   // 加载内核源码
    std::ifstream file("/root/projects/cuda_practice/linear/ocl/linear_v0.cl");
    // if (!file.is_open()) {
    //     std::cerr << "错误：无法打开内核文件！" << std::endl;
    //     throw std::runtime_error("Failed to open kernel file");
    // }

    std::string kernel_code(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );
    file.close();

    // if (kernel_code.empty()) {
    //     std::cerr << "错误：内核文件内容为空！" << std::endl;
    //     throw std::runtime_error("Kernel file is empty");
    // }

    // 创建程序
    const char* src = kernel_code.c_str();
    cl_int err;
    cl_program program = clCreateProgramWithSource(env.context, 1, &src, nullptr, &err);
    CHECK_CL_ERROR(err, "Failed to create program");

    // 构建程序
    err = clBuildProgram(program, 1, &env.device, nullptr, nullptr, nullptr);
    // if (err != CL_SUCCESS) {
    //     // 获取构建日志
    //     size_t log_size;
    //     clGetProgramBuildInfo(program, env.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    //     std::vector<char> log(log_size);
    //     clGetProgramBuildInfo(program, env.device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
    //     std::cerr << "Program build error: " << std::string(log.data(), log_size) << std::endl;
    //     CHECK_CL_ERROR(err, "Program build failed");
    // }

    // 创建内核
    cl_kernel kernel = clCreateKernel(program, "linear_kernel", &err);
    CHECK_CL_ERROR(err, "Failed to create test_kernel");

    // 设置内核参数
    cl_mem input_mem = input_buf;
    cl_mem weight_mem = weight_buf;
    cl_mem bias_mem = bias_buf;
    cl_mem output_mem = output_buf;
    //也可以传入
    CHECK_CL_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buf), "Arg 0 failed");
    CHECK_CL_ERROR(clSetKernelArg(kernel, 1, sizeof(cl_mem), &weight_mem), "Arg 1 failed");
    CHECK_CL_ERROR(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_mem), "Arg 2 failed");
    CHECK_CL_ERROR(clSetKernelArg(kernel, 3, sizeof(cl_mem), &output_mem), "Arg 3 failed");
    CHECK_CL_ERROR(clSetKernelArg(kernel, 4, sizeof(int), &batch_size), "Arg 4 failed");
    CHECK_CL_ERROR(clSetKernelArg(kernel, 5, sizeof(int), &input_features), "Arg 5 failed");
    CHECK_CL_ERROR(clSetKernelArg(kernel, 6, sizeof(int), &output_features), "Arg 6 failed");
    // 执行内核
    const size_t global[2] = {batch_size, output_features};
    const size_t local[2] = {16, 16};
    CHECK_CL_ERROR(clEnqueueNDRangeKernel(env.queue, kernel, 2, nullptr, global, local, 
                                        0, nullptr, nullptr),
                 "Kernel execution failed");

    // 读取结果
    CHECK_CL_ERROR(clEnqueueReadBuffer(env.queue, output_buf, CL_TRUE, 0,
                                    output.size() * sizeof(float), output.data(),
                                    0, nullptr, nullptr),
                 "Read buffer failed");

    std::cout << "First output value: " << output[0] << std::endl;

    // 清理资源
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    return 0;
} catch (const std::exception& e) {
    std::cerr << "Fatal Error: " << e.what() << std::endl;
    return 1;
}