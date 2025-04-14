/******************************************************************************
 * MIT License
 * 
 * Original work Copyright (c) 2023 Bruce-Lee-LY
 * https://github.com/Bruce-Lee-LY/cuda_hgemm/commit/0d26c2e4415ab0d0af5a6bfae301275c608d46b4
 * 
 * Modified work Copyright (c) 2025 xiaofanshao
 * 
 ******************************************************************************/

#pragma once

#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

inline char *curr_time() {
    time_t raw_time = time(nullptr);
    struct tm *time_info = localtime(&raw_time);
    static char now_time[64];
    now_time[strftime(now_time, sizeof(now_time), "%Y-%m-%d %H:%M:%S", time_info)] = '\0';

    return now_time;
}

inline int get_pid() {
    static int pid = getpid();

    return pid;
}

inline long int get_tid() {
    thread_local long int tid = syscall(SYS_gettid);

    return tid;
}

#define HGEMM_LOG_TAG "HGEMM"
#define HGEMM_LOG_FILE(x) (strrchr(x, '/') ? (strrchr(x, '/') + 1) : x)
#define HLOG(format, ...)                                                                                         \
    do {                                                                                                          \
        fprintf(stderr, "[%s %s %d:%ld %s:%d %s] " format "\n", HGEMM_LOG_TAG, curr_time(), get_pid(), get_tid(), \
                HGEMM_LOG_FILE(__FILE__), __LINE__, __FUNCTION__, ##__VA_ARGS__);                                 \
    } while (0)
