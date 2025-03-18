#include "../include/utils.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>

Timer::Timer() : is_running(false) {}

void Timer::start() {
    start_time = std::chrono::high_resolution_clock::now();
    is_running = true;
}

void Timer::stop() {
    end_time = std::chrono::high_resolution_clock::now();
    is_running = false;
}

double Timer::elapsed_milliseconds() {
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    
    if (is_running) {
        end = std::chrono::high_resolution_clock::now();
    } else {
        end = end_time;
    }
    
    return std::chrono::duration<double, std::milli>(end - start_time).count();
}

double Timer::elapsed_seconds() {
    return elapsed_milliseconds() / 1000.0;
}

void print_cuda_device_info() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cout << "No CUDA devices found!" << std::endl;
        return;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s):" << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max thread dimensions: (" 
                  << deviceProp.maxThreadsDim[0] << ", " 
                  << deviceProp.maxThreadsDim[1] << ", " 
                  << deviceProp.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max grid dimensions: (" 
                  << deviceProp.maxGridSize[0] << ", " 
                  << deviceProp.maxGridSize[1] << ", " 
                  << deviceProp.maxGridSize[2] << ")" << std::endl;
        std::cout << "  Warp size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Clock rate: " << deviceProp.clockRate / 1000 << " MHz" << std::endl;
        std::cout << std::endl;
    }
}

// 生成凯撒窗口函数
void generate_kaiser_window(float* window, int size, float beta) {
    // 贝塞尔函数I0的近似计算
    auto bessel_i0 = [](float x) {
        float sum = 1.0f;
        float term = 1.0f;
        
        for (int i = 1; i <= 20; i++) {
            float xi = x / 2.0f;
            term *= (xi * xi) / (i * i);
            sum += term;
            if (term < 1e-10f * sum) break;
        }
        
        return sum;
    };
    
    float i0_beta = bessel_i0(beta);
    
    for (int i = 0; i < size; i++) {
        float x = 2.0f * i / (size - 1.0f) - 1.0f;
        float arg = beta * std::sqrt(1.0f - x * x);
        window[i] = bessel_i0(arg) / i0_beta;
    }
}

// 为复数向量分配内存
void allocate_complex_vector(cufftComplex** vec, size_t size) {
    cudaMalloc((void**)vec, size * sizeof(cufftComplex));
}

// 释放复数向量
void free_complex_vector(cufftComplex* vec) {
    if (vec) {
        cudaFree(vec);
    }
}

// 在主机和设备间复制复数向量
void copy_complex_vector_host_to_device(cufftComplex* d_vec, const cufftComplex* h_vec, size_t size) {
    cudaMemcpy(d_vec, h_vec, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
}

void copy_complex_vector_device_to_host(cufftComplex* h_vec, const cufftComplex* d_vec, size_t size) {
    cudaMemcpy(h_vec, d_vec, size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
}

// 创建设备数组的模板实现
template <typename T>
void create_device_array(T** d_array, size_t size) {
    cudaMalloc((void**)d_array, size * sizeof(T));
}

template void create_device_array<float>(float** d_array, size_t size);
template void create_device_array<cufftComplex>(cufftComplex** d_array, size_t size);
template void create_device_array<int>(int** d_array, size_t size);

// 销毁设备数组的模板实现
template <typename T>
void destroy_device_array(T* d_array) {
    if (d_array) {
        cudaFree(d_array);
    }
}

template void destroy_device_array<float>(float* d_array);
template void destroy_device_array<cufftComplex>(cufftComplex* d_array);
template void destroy_device_array<int>(int* d_array);

// 保存矩阵到CSV文件
void save_matrix_to_csv(const std::string& filename, const float* data, int rows, int cols) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            file << data[i * cols + j];
            if (j < cols - 1) file << ",";
        }
        file << std::endl;
    }
    
    file.close();
}

// 归一化矩阵
void normalize_matrix(float* matrix, int rows, int cols) {
    float max_val = 0.0f;
    
    // 找出最大值
    for (int i = 0; i < rows * cols; i++) {
        max_val = std::max(max_val, std::abs(matrix[i]));
    }
    
    // 如果最大值不为零，则归一化
    if (max_val > 1e-10f) {
        for (int i = 0; i < rows * cols; i++) {
            matrix[i] /= max_val;
        }
    }
}

// 计算高斯权重
float gaussian_weight(float x, float max_x) {
    return std::exp(-std::pow(x / max_x, 2));
}

// 计算熵权重
float entropy_weight(float* data, int size) {
    float sum = 0.0f;
    
    // 计算总和
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    
    // 如果总和接近零，返回0
    if (sum < 1e-10f) {
        return 0.0f;
    }
    
    // 计算概率分布
    std::vector<float> prob(size);
    for (int i = 0; i < size; i++) {
        prob[i] = data[i] / sum;
    }
    
    // 计算熵
    float entropy = 0.0f;
    for (int i = 0; i < size; i++) {
        if (prob[i] > 1e-10f) {
            entropy -= prob[i] * std::log(prob[i]);
        }
    }
    
    return entropy;
}