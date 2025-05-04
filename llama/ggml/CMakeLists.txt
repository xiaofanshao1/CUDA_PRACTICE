# simple-gemm

set(TEST_TARGET simple-sgemm)

# 定义选项，默认启用 CUDA
option(GGML_CUDA "Enable CUDA support" ON)


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
                -DGGML_CUDA=${GGML_CUDA} 
                -DCMAKE_INSTALL_PREFIX=${GGML_INSTALL_DIR}
            INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
        )

    endif()
endif()


add_executable(${TEST_TARGET} simple_gemm.cpp)
add_dependencies(${TEST_TARGET} ggml_project)

target_include_directories(${TEST_TARGET} PRIVATE "${GGML_INSTALL_DIR}/include")
target_link_directories(${TEST_TARGET} PRIVATE "${GGML_INSTALL_DIR}/lib")
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml-cpu ggml-base ggml-cuda)
