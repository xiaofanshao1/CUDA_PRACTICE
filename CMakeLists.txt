cmake_minimum_required(VERSION 3.20.0)
project(cuda_basic_pattern VERSION 0.1.0 LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# 1. cmake有哪些重大的版本更新？
#    - CMake 3.0: 引入了新的语法和特性，如 target-based 构建系统。
#    - CMake 3.5: 支持 CUDA 作为一等公民。
#    - CMake 3.8: 引入了 `FindCUDA` 的替代品 `CUDA` 模块。
#    - CMake 3.15: 改进了 CUDA 支持，如 `CUDA_ARCHITECTURES`。
#    - CMake 3.20: 支持更多现代 C++ 特性和 CUDA 工具链。

# 2. 大写似乎能识别这个是怎么回事？LANGUAGE 这种如何识别？列表？
#    - CMake 不区分大小写，但推荐使用大写以提高可读性。
#    - `LANGUAGES` 是一个关键字，用于指定项目的编程语言。
#    - 语言列表可以是 `C`、`CXX`、`CUDA` 等，用空格分隔。

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES "86")  # 计算能力8.6

#add_subdirectory(common)
#add_subdirectory(reduce)
#add_subdirectory(sgemm)

#add_subdirectory(attentin)
#add_subdirectory(linear)
add_subdirectory(transpose)
#add_subdirectory(llama)

