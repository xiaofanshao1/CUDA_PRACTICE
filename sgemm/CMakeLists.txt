message(STATUS ">>> CUDA ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")  # 必须显示86


add_executable(sgemm_v0_global_mem sgemm_v0_global_mem.cu)
add_executable(sgemm_v1_shared_mem sgemm_v1_shared_mem.cu)
add_executable(sgemm_v2_shared_mem_splitk sgemm_v2_shared_mem_splitk.cu)
add_executable(sgemm_v3_increase_work_of_per_thread sgemm_v3_increase_work_of_per_thread.cu)
add_executable(sgemm_v4_float4_loading sgemm_v4_float4_loading.cu)
add_executable(sgemm_v5_register_tile_outer_product sgemm_v5_register_tile_outer_product.cu)
add_executable(sgemm_v6_register_tile_outer_product_float4 sgemm_v6_register_tile_outer_product_float4.cu)
add_executable(sgemm_v7_register_tile_outer_product_float4_transpose sgemm_v7_register_tile_outer_product_float4_transpose.cu)
add_executable(sgemm_v8_double_buffer sgemm_v8_double_buffer.cu)