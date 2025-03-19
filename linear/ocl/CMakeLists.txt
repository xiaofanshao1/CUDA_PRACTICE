# ==========================================================
# 基于icd dispatcher的方式需要配置，需要
# /etc/OpenCL/vendors/nvidia.icd: No such file or directory
# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# find_package(OpenCL REQUIRED)
# include_directories(${OpenCL_INCLUDE_DIRS})

# add_executable(linear_ocl_v0 main.cpp)
# target_link_libraries(linear_ocl_v0 
#     PRIVATE ${OpenCL_LIBRARIES}
# )
# ==========================================================


# 添加了 /etc/OpenCL/vendors/nvidia.icd后，clinfo显示正常
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_executable(linear_ocl_v0 main.cpp)
target_include_directories(linear_ocl_v0 PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(linear_ocl_v0 PRIVATE
    ${CUDAToolkit_LIBRARY_DIR}/libOpenCL.so.1  # 显式指定动态库路径
)

set_target_properties(linear_ocl_v0 PROPERTIES
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

