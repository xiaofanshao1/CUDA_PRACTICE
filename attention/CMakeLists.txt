project(attention)

set(LIBTORCH_ROOT "${CMAKE_SOURCE_DIR}/thirdparty/libtorch")
#set(LIBTORCH_ROOT "/root/projects/cuda_practice/thirdparty/libtorch")
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")

find_package(Torch REQUIRED)
# 添加CUDA支持
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 添加可执行文件
add_executable(attention_v0 
    attention_torch_v0.cpp  # 如果是CUDA代码需要改成.cu扩展名
)

# 链接Torch库
target_link_libraries(attention_v0
  PRIVATE 
    "${TORCH_LIBRARIES}"
)

