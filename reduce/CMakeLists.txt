add_executable(reduce_v0_global_mem reduce_v0_global_mem.cu)
add_executable(reduce_v1_shared_mem reduce_v1_shared_mem.cu)
add_executable(reduce_v2_warp_divergence reduce_v2_warp_divergence.cu)
add_executable(reduce_v3_no_bank_conflict reduce_v3_no_bank_conflict.cu)
add_executable(reduce_v4_block_utilization reduce_v4_block_utilization.cu)
add_executable(reduce_v5_unroll_last_warp reduce_v5_unroll_last_warp.cu)
add_executable(reduce_v6_unroll_all_warp reduce_v6_unroll_all_warp.cu)
add_executable(reduce_v7_multi_add reduce_v7_multi_add.cu) 
add_executable(reduce_v8_warp_shuffle reduce_v8_warp_shuffle.cu)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_v8_warp_shuffle PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

#问题：如果链接cuda runtime 和其他库怎么做?
# target_link_libraries(reduce_v4_block_utilization PRIVATE CUDA::cudart CUDA::cublas)
# 或者 ${CUDA_cublas_LIBRARY} 需要自己解决依赖，比如联入runtime

#问题：如果设置debug cuda的模式？
