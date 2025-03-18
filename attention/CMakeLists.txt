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
add_executable(libtorch_test 
libtorch_test.cpp.cpp  # 如果是CUDA代码需要改成.cu扩展名
)
# 链接Torch库
target_link_libraries(libtorch_test
  PRIVATE 
    "${TORCH_LIBRARIES}"
)

add_executable(flash_attn flash_attn_main.cpp flash_attn.cu)
target_link_libraries(flash_attn PRIVATE "${TORCH_LIBRARIES}")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(flash_attn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
