# simple-gemm

set(TEST_TARGET simple-sgemm)

# 定义选项，默认启用 CUDA
if(USE_CUDA)
    option(GGML_CUDA "Enable CUDA support" ON)
endif()



set(GGML_DIR "${CMAKE_SOURCE_DIR}/thirdparty/ggml")
# 定义 GGML 的安装路径（通常在 build/ggml_install 目录）
set(GGML_INSTALL_DIR "${CMAKE_BINARY_DIR}/ggml_install")

if(NOT TARGET ggml)
    if(NOT EXISTS ${GGML_DIR}/CMakeLists.txt)
        message(WARNING "GGML submodule not initialized. Run: git submodule update --init")
    else()
        include(ExternalProject)
        ExternalProject_Add(
            ggml_project
            SOURCE_DIR ${GGML_DIR}
            BINARY_DIR "${CMAKE_BINARY_DIR}/ggml_build"
            CMAKE_ARGS 
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}  # 继承主项目的构建类型
                -DGGML_CUDA=${GGML_CUDA} 
                -DCMAKE_INSTALL_PREFIX=${GGML_INSTALL_DIR}
            INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
        )

    endif()
endif()


add_executable(${TEST_TARGET} simple_gemm.cpp)
add_dependencies(${TEST_TARGET} ggml_project)

target_include_directories(${TEST_TARGET} PRIVATE "${GGML_INSTALL_DIR}/include")


# 根据构建类型选择库路径
target_link_directories(${TEST_TARGET} PRIVATE 
    "${GGML_INSTALL_DIR}/lib/$<IF:$<CONFIG:Debug>,debug,>"
    "${GGML_INSTALL_DIR}/lib"
)

target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml-cpu ggml-base $<$<BOOL:${GGML_CUDA}>:ggml-cuda>)

# 统一配置调试选项（支持混合 CUDA + CPU 程序）
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # CPU 代码：禁用优化并生成符号表
    target_compile_options(${TEST_TARGET} PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:-g -O0>
    )
    # CUDA 代码：生成设备调试符号
    target_compile_options(${TEST_TARGET} PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-G 00 >
    )
    # 链接阶段统一保留符号
    target_link_options(${TEST_TARGET} PRIVATE -g)
endif()