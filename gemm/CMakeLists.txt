message(STATUS ">>> CUDA ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")  # 必须显示86


add_executable(sgemm_v0_global_mem sgemm_v0_global_mem.cu)
add_executable(sgemm_v1_shared_mem sgemm_v1_shared_mem.cu)
add_executable(sgemm_v2_shared_mem_splitk sgemm_v2_shared_mem_splitk.cu)
add_executable(sgemm_v3_increase_work_of_per_thread sgemm_v3_increase_work_of_per_thread.cu)
