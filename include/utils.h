#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <cufft.h>
#include <cuda_runtime.h>

// CUDA错误检查宏
#define CHECK_CUDA_ERROR(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// cuFFT错误检查宏
#define CHECK_CUFFT_ERROR(err) do { \
    if (err != CUFFT_SUCCESS) { \
        fprintf(stderr, "CUFFT Error at %s:%d - Code %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 计时器类，用于性能测量
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    bool is_running;

public:
    Timer();
    void start();
    void stop();
    double elapsed_milliseconds();
    double elapsed_seconds();
};

// 打印CUDA设备信息
void print_cuda_device_info();

// 计算凯撒窗口函数
void generate_kaiser_window(float* window, int size, float beta);

// 为复数向量分配内存
void allocate_complex_vector(cufftComplex** vec, size_t size);

// 释放复数向量
void free_complex_vector(cufftComplex* vec);

// 在主机和设备间复制复数向量
void copy_complex_vector_host_to_device(cufftComplex* d_vec, const cufftComplex* h_vec, size_t size);
void copy_complex_vector_device_to_host(cufftComplex* h_vec, const cufftComplex* d_vec, size_t size);

// 创建和销毁设备数组
template <typename T>
void create_device_array(T** d_array, size_t size);

template <typename T>
void destroy_device_array(T* d_array);

// 保存数据到文件
void save_matrix_to_csv(const std::string& filename, const float* data, int rows, int cols);

// 归一化矩阵
void normalize_matrix(float* matrix, int rows, int cols);

// 数学函数
float gaussian_weight(float x, float max_x);
float entropy_weight(float* data, int size);

#endif // UTILS_H
