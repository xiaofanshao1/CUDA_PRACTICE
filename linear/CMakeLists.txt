# project(linear)

# set(LIBTORCH_ROOT "${CMAKE_SOURCE_DIR}/thirdparty/libtorch")
# list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")

# find_package(Torch REQUIRED)

# enable_language(CUDA)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# add_executable(linear_v0
#     linear_v0.cu  # 如果是CUDA代码需要改成.cu扩展名
# )

# target_link_libraries(attention_v0
#   PRIVATE 
#     "${TORCH_LIBRARIES}"
# )
add_subdirectory(ocl)