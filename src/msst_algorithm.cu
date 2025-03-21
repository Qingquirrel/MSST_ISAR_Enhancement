#include "../include/msst_algorithm.h"
#include "../include/utils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cfloat>  // 或者使用 #include <float.h>

// 调试辅助函数：检查输入数据统计信息
void checkComplexData(const cufftComplex* data, int size, const char* stage_name) {
    float max_mag = 0.0f, min_mag = FLT_MAX, sum_mag = 0.0f;
    int zeros_count = 0;
    
    for (int i = 0; i < size; i++) {
        float mag = sqrtf(data[i].x * data[i].x + data[i].y * data[i].y);
        max_mag = std::max(max_mag, mag);
        min_mag = std::min(min_mag, mag);
        sum_mag += mag;
        if (mag < 1e-10f) zeros_count++;
    }
    
    std::cout << stage_name << " 数据统计: "
              << "最大值=" << max_mag 
              << ", 最小值=" << min_mag 
              << ", 平均值=" << (sum_mag / size)
              << ", 零值数量=" << zeros_count 
              << " (占比 " << (float)zeros_count/size*100.0f << "%)" << std::endl;
}

// 调试辅助函数：检查实数数据统计信息
void checkFloatData(const float* data, int size, const char* stage_name) {
    float max_val = 0.0f, min_val = FLT_MAX, sum_val = 0.0f;
    int zeros_count = 0;
    
    for (int i = 0; i < size; i++) {
        max_val = std::max(max_val, data[i]);
        min_val = std::min(min_val, data[i]);
        sum_val += data[i];
        if (fabs(data[i]) < 1e-10f) zeros_count++;
    }
    
    std::cout << stage_name << " 数据统计: "
              << "最大值=" << max_val 
              << ", 最小值=" << min_val 
              << ", 平均值=" << (sum_val / size)
              << ", 零值数量=" << zeros_count 
              << " (占比 " << (float)zeros_count/size*100.0f << "%)" << std::endl;
}

// 调试辅助函数：保存复数数据到文件
void saveComplexDataToFile(const cufftComplex* data, int size, const char* filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }
    file.write(reinterpret_cast<const char*>(data), size * sizeof(cufftComplex));
    file.close();
    std::cout << "已保存数据到文件: " << filename << std::endl;
}

// 调试辅助函数：保存实数数据到文件
void saveFloatDataToFile(const float* data, int size, const char* filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }
    file.write(reinterpret_cast<const char*>(data), size * sizeof(float));
    file.close();
    std::cout << "已保存数据到文件: " << filename << std::endl;
}

// CUDA核心函数：应用Kaiser窗口
__global__ void applyKaiserWindowKernel(const cufftComplex* input, cufftComplex* output, 
                                      const float* window, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx].x = input[idx].x * window[idx];
        output[idx].y = input[idx].y * window[idx];
    }
}

// CUDA核心函数：生成二维Kaiser窗口
__global__ void generate2DKaiserWindowKernel(float* window, const float* win_burst, 
                                          const float* win_pulses, int burst, int pulses) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < burst && col < pulses) {
        int idx = row * pulses + col;
        window[idx] = win_burst[row] * win_pulses[col];
    }
}

// CUDA核心函数：展平矩阵
__global__ void flattenMatrixKernel(const cufftComplex* input, cufftComplex* output, 
                                 int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < height * width) {
        int row = idx / width;
        int col = idx % width;
        output[idx] = input[row * width + col];
    }
}

// CUDA核心函数：计算Morlet小波
__global__ void morletWaveletKernel(cufftComplex* wavelet, float scale, float centerFreq, 
                                  float* t, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float t_scaled = t[idx] / scale;
        float gaussian = expf(-(t_scaled * t_scaled) / 2.0f);
        float angle = 2.0f * M_PI * centerFreq * t_scaled;
        float sinVal, cosVal;
        sincosf(angle, &sinVal, &cosVal);
        wavelet[idx].x = gaussian * cosVal;
        wavelet[idx].y = gaussian * sinVal;
    }
}

// CUDA核心函数：复数卷积（使用直接卷积而不是频域乘法）
__global__ void complexConvolveSameKernel(const cufftComplex* signal, const cufftComplex* kernel,
                                       cufftComplex* result, int signal_size, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < signal_size) {
        float real_sum = 0.0f;
        float imag_sum = 0.0f;
        int half_kernel = kernel_size / 2;
        
        for (int k = 0; k < kernel_size; k++) {
            int sig_idx = idx - half_kernel + k;
            if (sig_idx >= 0 && sig_idx < signal_size) {
                real_sum += signal[sig_idx].x * kernel[k].x - signal[sig_idx].y * kernel[k].y;
                imag_sum += signal[sig_idx].x * kernel[k].y + signal[sig_idx].y * kernel[k].x;
            }
        }
        
        result[idx].x = real_sum;
        result[idx].y = imag_sum;
    }
}

// CUDA核心函数：复数向量的绝对值
__global__ void complexAbsKernel(const cufftComplex* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = sqrtf(input[idx].x * input[idx].x + input[idx].y * input[idx].y);
    }
}

// CUDA核心函数：计算概率分布和局部熵
__global__ void calculateEntropyKernel(const float* input, float* output, float* local_entropy,
                                    int size) {
    // 先计算总和用于归一化
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += input[i];
    }
    
    // 归一化为概率分布并计算熵
    float entropy = 0.0f;
    if (sum > 1e-10f) {
        for (int i = 0; i < size; i++) {
            float prob = input[i] / sum;
            if (prob > 1e-10f) {
                entropy -= prob * logf(prob);
            }
        }
    }
    
    // 应用熵优化
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * expf(-entropy);
    }
    
    // 保存局部熵值（只需要一个线程写入）
    if (idx == 0) {
        *local_entropy = entropy;
    }
}

// CUDA核心函数：应用熵优化
__global__ void applyEntropyOptimizationKernel(float* data, float entropy, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * expf(-entropy);
    }
}

// CUDA核心函数：计算方差
__global__ void calculateVarianceKernel(const float* data, float* variance, float mean, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        float sum_sq_diff = 0.0f;
        for (int i = 0; i < size; i++) {
            float diff = data[i] - mean;
            sum_sq_diff += diff * diff;
        }
        *variance = sum_sq_diff / size;
    }
}

// CUDA核心函数：融合多尺度结果
__global__ void fuseScalesKernel(const float* sst_image, float* fused_sst, 
                               const float* scale_weights, int num_scales, int signal_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < signal_length) {
        float sum = 0.0f;
        for (int s = 0; s < num_scales; s++) {
            sum += scale_weights[s] * sst_image[s * signal_length + idx];
        }
        fused_sst[idx] = sum;
    }
}

// CUDA核心函数：非线性拉伸和增益
__global__ void nonLinearStretchAndGainKernel(float* image, float power, float gain, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // 应用非线性拉伸
        image[idx] = powf(image[idx], power);
        
        // 应用增益
        image[idx] = image[idx] * gain;
        
        // 限幅
        if (image[idx] > 1.0f) {
            image[idx] = 1.0f;
        }
    }
}

// CUDA核心函数：执行FFT移位
__global__ void fftshiftKernel(cufftComplex* input, float* output, int burst, int pulses) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < pulses && y < burst) {
        int src_idx = y * pulses + x;
        
        // 计算移位后的坐标
        int shift_x = (x + pulses/2) % pulses;
        int shift_y = (y + burst/2) % burst;
        int dst_idx = shift_y * pulses + shift_x;
        
        // 计算复数幅度
        float real = input[src_idx].x;
        float imag = input[src_idx].y;
        float magnitude = sqrtf(real * real + imag * imag);
        
        // 存储结果
        output[dst_idx] = magnitude;
    }
}

// CUDA核心函数：转换为dB刻度
__global__ void convertToDbKernel(const float* input, float* output, float min_db, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float value = input[idx];
        if (value > 1e-10f) {
            output[idx] = 20.0f * log10f(value);
        } else {
            output[idx] = min_db;
        }
    }
}

// 窗口函数应用
void applyKaiserWindow(cufftComplex* h_Rx, cufftComplex* h_Rxw, int burst, int pulses, float beta) {
    // 计算数据总大小
    int size = burst * pulses;
    
    // 打印处理信息
    std::cout << "应用Kaiser窗口函数: " << burst << "x" << pulses << " 矩阵" << std::endl;
    
    try {
        // 检查输入数据
        checkComplexData(h_Rx, size, "Kaiser窗口前");
        
        // 在主机上生成Kaiser窗口
        float* h_win_burst = new float[burst];
        float* h_win_pulses = new float[pulses];
        
        // 生成各维度的Kaiser窗口
        generate_kaiser_window(h_win_burst, burst, beta);
        generate_kaiser_window(h_win_pulses, pulses, beta);
        
        // 在设备上分配内存
        cufftComplex *d_Rx, *d_Rxw;
        float *d_win_burst, *d_win_pulses, *d_window;
        
        cudaMalloc(&d_Rx, size * sizeof(cufftComplex));
        cudaMalloc(&d_Rxw, size * sizeof(cufftComplex));
        cudaMalloc(&d_win_burst, burst * sizeof(float));
        cudaMalloc(&d_win_pulses, pulses * sizeof(float));
        cudaMalloc(&d_window, size * sizeof(float));
        
        // 将数据从主机复制到设备
        cudaMemcpy(d_Rx, h_Rx, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
        cudaMemcpy(d_win_burst, h_win_burst, burst * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_win_pulses, h_win_pulses, pulses * sizeof(float), cudaMemcpyHostToDevice);
        
        // 生成2D窗口
        dim3 blockDim(16, 16);
        dim3 gridDim((pulses + blockDim.x - 1) / blockDim.x, 
                     (burst + blockDim.y - 1) / blockDim.y);
        
        generate2DKaiserWindowKernel<<<gridDim, blockDim>>>(d_window, d_win_burst, d_win_pulses, burst, pulses);
        cudaDeviceSynchronize();
        
        // 应用窗口
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        
        applyKaiserWindowKernel<<<numBlocks, blockSize>>>(d_Rx, d_Rxw, d_window, size);
        cudaDeviceSynchronize();
        
        // 将结果从设备复制回主机
        cudaMemcpy(h_Rxw, d_Rxw, size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
        
        // 检查输出数据
        checkComplexData(h_Rxw, size, "Kaiser窗口后");
        
        // 清理设备内存
        cudaFree(d_Rx);
        cudaFree(d_Rxw);
        cudaFree(d_win_burst);
        cudaFree(d_win_pulses);
        cudaFree(d_window);
        
        // 清理主机内存
        delete[] h_win_burst;
        delete[] h_win_pulses;
        
        std::cout << "Kaiser窗口应用完成" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "应用Kaiser窗口时发生错误: " << e.what() << std::endl;
        throw;
    }
}

// 执行逆FFT
void performIFFT(cufftComplex* h_Rxw, cufftComplex* h_Es_IFFT, int burst, int pulses) {
    int size = burst * pulses;
    
    // 检查输入数据
    checkComplexData(h_Rxw, size, "IFFT前");
    
    try {
        // 分配设备内存
        cufftComplex *d_Rxw, *d_Es_IFFT;
        cudaMalloc(&d_Rxw, size * sizeof(cufftComplex));
        cudaMalloc(&d_Es_IFFT, size * sizeof(cufftComplex));
        
        // 将数据从主机复制到设备
        cudaMemcpy(d_Rxw, h_Rxw, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
        
        // 创建CUFFT计划
        cufftHandle plan;
        cufftPlan1d(&plan, size, CUFFT_C2C, 1);
        
        // 执行逆FFT
        cufftExecC2C(plan, d_Rxw, d_Es_IFFT, CUFFT_INVERSE);
        
        // 归一化IFFT结果
        float scale = 1.0f / size;
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSscal(handle, 2 * size, &scale, (float*)d_Es_IFFT, 1);
        cublasDestroy(handle);
        
        // 将结果从设备复制回主机
        cudaMemcpy(h_Es_IFFT, d_Es_IFFT, size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
        
        // 检查输出数据
        checkComplexData(h_Es_IFFT, size, "IFFT后");
        
        // 清理
        cufftDestroy(plan);
        cudaFree(d_Rxw);
        cudaFree(d_Es_IFFT);
        
        std::cout << "IFFT执行完成" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "执行IFFT时发生错误: " << e.what() << std::endl;
        throw;
    }
}

// 计算Morlet小波变换
void calculateMorletWavelet(cufftComplex* h_Es_IFFT, float* h_SST_Image, 
                           float* h_scale_weights, const MsstParams& params) {
    int burst = params.burst;
    int pulses = params.pulses;
    int size = burst * pulses;
    int num_scales = params.num_scales;
    
    std::cout << "计算Morlet小波变换，尺度数量: " << num_scales << std::endl;
    
    try {
        // 检查输入数据
        checkComplexData(h_Es_IFFT, size, "小波变换前");
        
        // 分配设备内存
        cufftComplex *d_Es_IFFT, *d_wavelet, *d_convolved;
        float *d_t, *d_abs_convolved, *d_sst_row, *d_entropy;
        
        cudaMalloc(&d_Es_IFFT, size * sizeof(cufftComplex));
        cudaMalloc(&d_wavelet, size * sizeof(cufftComplex));
        cudaMalloc(&d_convolved, size * sizeof(cufftComplex));
        cudaMalloc(&d_t, size * sizeof(float));
        cudaMalloc(&d_abs_convolved, size * sizeof(float));
        cudaMalloc(&d_sst_row, size * sizeof(float));
        cudaMalloc(&d_entropy, sizeof(float));
        
        // 将Es_IFFT复制到设备
        cudaMemcpy(d_Es_IFFT, h_Es_IFFT, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
        
        // 创建时间向量
        float* h_t = new float[size];
        for (int i = 0; i < size; i++) {
            h_t[i] = 2.0f * i / (size - 1.0f) - 1.0f;  // 范围从-1到1
        }
        cudaMemcpy(d_t, h_t, size * sizeof(float), cudaMemcpyHostToDevice);
        
        // 设置内核参数
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        
        // 为每个尺度生成Morlet小波并计算变换
        for (int s = 0; s < num_scales; s++) {
            // 计算当前尺度（对应MATLAB中的scales = linspace(1, 128, 512)）
            float scale = params.min_scale + (params.max_scale - params.min_scale) * s / (num_scales - 1);
            
            // 生成Morlet小波
            morletWaveletKernel<<<numBlocks, blockSize>>>(d_wavelet, scale, params.f0, d_t, size);
            cudaDeviceSynchronize();
            
            // 执行卷积（使用直接卷积替代频域乘法）
            complexConvolveSameKernel<<<numBlocks, blockSize>>>(d_Es_IFFT, d_wavelet, d_convolved, size, size);
            cudaDeviceSynchronize();
            
            // 计算绝对值（对应MATLAB中的abs(conv(...))）
            complexAbsKernel<<<numBlocks, blockSize>>>(d_convolved, d_abs_convolved, size);
            cudaDeviceSynchronize();
            
            // 计算加权和熵优化（对应MATLAB代码中的权重和熵计算）
            // 计算局部熵
            calculateEntropyKernel<<<numBlocks, blockSize>>>(d_abs_convolved, d_sst_row, d_entropy, size);
            cudaDeviceSynchronize();
            
            // 将结果复制到SST图像
            cudaMemcpy(&h_SST_Image[s * size], d_sst_row, size * sizeof(float), cudaMemcpyDeviceToHost);
            
            // 计算当前尺度的权重（基于方差的倒数，对应MATLAB中的1/(1+var(...))）
            float h_entropy;
            cudaMemcpy(&h_entropy, d_entropy, sizeof(float), cudaMemcpyDeviceToHost);
            
            // 计算平均值和方差
            float mean = 0.0f;
            for (int i = 0; i < size; i++) {
                mean += h_SST_Image[s * size + i];
            }
            mean /= size;
            
            float variance = 0.0f;
            for (int i = 0; i < size; i++) {
                float diff = h_SST_Image[s * size + i] - mean;
                variance += diff * diff;
            }
            variance /= size;
            
            // 设置尺度权重（方差反比权重，对应MATLAB代码）
            h_scale_weights[s] = 1.0f / (1.0f + variance);
            
            if (s % 50 == 0 || s == num_scales - 1) {
                std::cout << "已完成尺度 " << s+1 << "/" << num_scales 
                          << ", 尺度=" << scale 
                          << ", 权重=" << h_scale_weights[s] 
                          << ", 熵=" << h_entropy << std::endl;
            }
        }
        
        // 规范化尺度权重
        float sum_weights = 0.0f;
        for (int s = 0; s < num_scales; s++) {
            sum_weights += h_scale_weights[s];
        }
        
        if (sum_weights > 1e-10f) {
            for (int s = 0; s < num_scales; s++) {
                h_scale_weights[s] /= sum_weights;
            }
        } else {
            // 如果权重和接近零，使用均匀权重
            for (int s = 0; s < num_scales; s++) {
                h_scale_weights[s] = 1.0f / num_scales;
            }
        }
        
        // 检查SST图像和权重
        float weights_sum = 0.0f;
        for (int s = 0; s < num_scales; s++) {
            weights_sum += h_scale_weights[s];
        }
        std::cout << "尺度权重总和: " << weights_sum << std::endl;
        
        // 保存部分中间结果
        saveFloatDataToFile(&h_SST_Image[0], size, "debug_sst_scale0.bin");
        saveFloatDataToFile(&h_SST_Image[(num_scales/2) * size], size, "debug_sst_scale_mid.bin");
        saveFloatDataToFile(h_scale_weights, num_scales, "debug_scale_weights.bin");
        
        // 清理
        cudaFree(d_Es_IFFT);
        cudaFree(d_wavelet);
        cudaFree(d_convolved);
        cudaFree(d_t);
        cudaFree(d_abs_convolved);
        cudaFree(d_sst_row);
        cudaFree(d_entropy);
        delete[] h_t;
        
        std::cout << "Morlet小波变换计算完成" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "计算Morlet小波变换时发生错误: " << e.what() << std::endl;
        throw;
    }
}

// 融合和增强SST图像
void fuseAndEnhanceSST(float* h_SST_Image, float* h_scale_weights, 
                      float* h_Enhanced_Rx, const MsstParams& params) {
    int burst = params.burst;
    int pulses = params.pulses;
    int size = burst * pulses;
    int num_scales = params.num_scales;
    
    std::cout << "融合和增强SST图像" << std::endl;
    
    try {
        // 分配设备内存
        float *d_sst_image, *d_scale_weights, *d_fused_sst;
        
        cudaMalloc(&d_sst_image, num_scales * size * sizeof(float));
        cudaMalloc(&d_scale_weights, num_scales * sizeof(float));
        cudaMalloc(&d_fused_sst, size * sizeof(float));
        
        // 复制数据到设备
        cudaMemcpy(d_sst_image, h_SST_Image, num_scales * size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_scale_weights, h_scale_weights, num_scales * sizeof(float), cudaMemcpyHostToDevice);
        
        // 设置内核参数
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        
        // 融合多尺度结果
        fuseScalesKernel<<<numBlocks, blockSize>>>(d_sst_image, d_fused_sst, d_scale_weights, num_scales, size);
        cudaDeviceSynchronize();
        
        // 将融合结果复制回主机
        cudaMemcpy(h_Enhanced_Rx, d_fused_sst, size * sizeof(float), cudaMemcpyDeviceToHost);
        
        // 检查融合后的数据
        checkFloatData(h_Enhanced_Rx, size, "融合后");
        
        // 非线性拉伸和增益
        nonLinearStretchAndGainKernel<<<numBlocks, blockSize>>>(d_fused_sst, params.non_linear_power, params.gain_factor, size);
        cudaDeviceSynchronize();
        
        // 将增强后的结果复制回主机
        cudaMemcpy(h_Enhanced_Rx, d_fused_sst, size * sizeof(float), cudaMemcpyDeviceToHost);
        
        // 检查增强后的数据
        checkFloatData(h_Enhanced_Rx, size, "非线性拉伸和增益后");
        
        // 保存增强后的数据
        saveFloatDataToFile(h_Enhanced_Rx, size, "debug_enhanced_rx.bin");
        
        // 清理设备内存
        cudaFree(d_sst_image);
        cudaFree(d_scale_weights);
        cudaFree(d_fused_sst);
        
        std::cout << "SST图像融合和增强完成" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "融合和增强SST图像时发生错误: " << e.what() << std::endl;
        throw;
    }
}

// 生成最终ISAR图像
void generateISARImage(float* h_Enhanced_Rx, float* h_Enhanced_ISAR, const MsstParams& params) {
    int burst = params.burst;
    int pulses = params.pulses;
    int size = burst * pulses;
    
    std::cout << "生成最终ISAR图像" << std::endl;
    
    try {
        // 检查输入数据
        checkFloatData(h_Enhanced_Rx, size, "ISAR生成前");
        
        // 分配设备内存
        cufftComplex *d_input, *d_output;
        float *d_magnitude, *d_enhanced_rx;
        
        cudaMalloc(&d_input, size * sizeof(cufftComplex));
        cudaMalloc(&d_output, size * sizeof(cufftComplex));
        cudaMalloc(&d_magnitude, size * sizeof(float));
        cudaMalloc(&d_enhanced_rx, size * sizeof(float));
        
        // 复制增强后的数据到设备
        cudaMemcpy(d_enhanced_rx, h_Enhanced_Rx, size * sizeof(float), cudaMemcpyHostToDevice);
        
        // 将实数数据转换为复数形式（虚部为零）
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        
        // 使用CUDA内核将实数转为复数
        // 或在主机上转换然后复制到设备
        cufftComplex* h_input = new cufftComplex[size];
        for (int i = 0; i < size; i++) {
            h_input[i].x = h_Enhanced_Rx[i];
            h_input[i].y = 0.0f;
        }
        
        // 复制到设备
        cudaMemcpy(d_input, h_input, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
        
        // 创建2D FFT计划
        cufftHandle plan;
        int dims[2] = {burst, pulses};
        cufftPlanMany(&plan, 2, dims, 
                     NULL, 1, size,  // 输入数据布局
                     NULL, 1, size,  // 输出数据布局
                     CUFFT_C2C, 1);  // 复数到复数
        
        // 执行FFT
        cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD);
        cudaDeviceSynchronize();
        
        // 执行FFT移位并计算幅度
        dim3 blockDim2D(16, 16);
        dim3 gridDim2D((pulses + blockDim2D.x - 1) / blockDim2D.x, 
                      (burst + blockDim2D.y - 1) / blockDim2D.y);
        
        fftshiftKernel<<<gridDim2D, blockDim2D>>>(d_output, d_magnitude, burst, pulses);
        cudaDeviceSynchronize();
        
        // 将结果复制回主机
        cudaMemcpy(h_Enhanced_ISAR, d_magnitude, size * sizeof(float), cudaMemcpyDeviceToHost);
        
        // 归一化ISAR图像
        float max_val = 0.0f;
        for (int i = 0; i < size; i++) {
            max_val = std::max(max_val, h_Enhanced_ISAR[i]);
        }
        
        if (max_val > 1e-10f) {
            for (int i = 0; i < size; i++) {
                h_Enhanced_ISAR[i] /= max_val;
            }
        } else {
            std::cerr << "警告: ISAR图像最大值接近零，归一化可能不正确" << std::endl;
        }
        
        // 检查生成的ISAR图像数据
        checkFloatData(h_Enhanced_ISAR, size, "ISAR生成后");
        
        // 保存ISAR图像数据
        saveFloatDataToFile(h_Enhanced_ISAR, size, "debug_enhanced_isar.bin");
        
        // 清理
        delete[] h_input;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_magnitude);
        cudaFree(d_enhanced_rx);
        cufftDestroy(plan);
        
        std::cout << "ISAR图像生成完成" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "生成ISAR图像时发生错误: " << e.what() << std::endl;
        throw;
    }
}

// MSST处理函数 - 主函数
void processMSST(cufftComplex* h_Rx, float* h_Enhanced_ISAR, const MsstParams& params) {
    int burst = params.burst;
    int pulses = params.pulses;
    int size = burst * pulses;
    
    std::cout << "开始MSST处理，矩阵大小: " << burst << "x" << pulses << std::endl;
    
    try {
        // 检查输入数据
        checkComplexData(h_Rx, size, "MSST处理前");
        
        // 保存输入数据用于调试
        saveComplexDataToFile(h_Rx, size, "debug_input_rx.bin");
        
        // 分配主机内存
        cufftComplex* h_Rxw = new cufftComplex[size];  // 加窗后的数据
        cufftComplex* h_Es_IFFT = new cufftComplex[size];  // IFFT后的数据
        float* h_SST_Image = new float[params.num_scales * size];  // SST图像
        float* h_scale_weights = new float[params.num_scales];  // 尺度权重
        float* h_Enhanced_Rx = new float[size];  // 增强后的数据
        
        // 初始化
        memset(h_Rxw, 0, size * sizeof(cufftComplex));
        memset(h_Es_IFFT, 0, size * sizeof(cufftComplex));
        memset(h_SST_Image, 0, params.num_scales * size * sizeof(float));
        memset(h_scale_weights, 0, params.num_scales * sizeof(float));
        memset(h_Enhanced_Rx, 0, size * sizeof(float));
        
        // 步骤1: 应用Kaiser窗口
        std::cout << "步骤1: 应用Kaiser窗口..." << std::endl;
        applyKaiserWindow(h_Rx, h_Rxw, burst, pulses, params.beta);
        
        // 保存加窗后的数据
        saveComplexDataToFile(h_Rxw, size, "debug_rxw.bin");
        
        // 步骤2: 执行逆FFT获取距离分布
        std::cout << "步骤2: 执行逆FFT获取距离分布..." << std::endl;
        performIFFT(h_Rxw, h_Es_IFFT, burst, pulses);
        
        // 保存IFFT后的数据
        saveComplexDataToFile(h_Es_IFFT, size, "debug_es_ifft.bin");
        
        // 步骤3: 多尺度小波变换
        std::cout << "步骤3: 多尺度小波变换分析..." << std::endl;
        calculateMorletWavelet(h_Es_IFFT, h_SST_Image, h_scale_weights, params);
        
        // 步骤4: 融合和增强SST图像
        std::cout << "步骤4: 融合和增强SST图像..." << std::endl;
        fuseAndEnhanceSST(h_SST_Image, h_scale_weights, h_Enhanced_Rx, params);
        
        // 步骤5: 生成最终ISAR图像
        std::cout << "步骤5: 生成最终ISAR图像..." << std::endl;
        generateISARImage(h_Enhanced_Rx, h_Enhanced_ISAR, params);
        
        // 最终状态检查
        checkFloatData(h_Enhanced_ISAR, size, "MSST处理最终结果");
        
        // 如果检测到最终数据全部为零，尝试替代方法
        bool all_zeros = true;
        for (int i = 0; i < size; i++) {
            if (h_Enhanced_ISAR[i] > 1e-10f) {
                all_zeros = false;
                break;
            }
        }
        
        if (all_zeros) {
            std::cout << "警告: 检测到最终ISAR数据全为零，尝试替代方法..." << std::endl;
            
            // 替代方法: 简单地使用输入数据的幅度作为输出
            for (int i = 0; i < size; i++) {
                h_Enhanced_ISAR[i] = sqrtf(h_Rx[i].x * h_Rx[i].x + h_Rx[i].y * h_Rx[i].y);
            }
            
            // 归一化
            float max_val = 0.0f;
            for (int i = 0; i < size; i++) {
                max_val = std::max(max_val, h_Enhanced_ISAR[i]);
            }
            
            if (max_val > 1e-10f) {
                for (int i = 0; i < size; i++) {
                    h_Enhanced_ISAR[i] /= max_val;
                }
            }
            
            // 检查替代输出
            checkFloatData(h_Enhanced_ISAR, size, "替代方法处理结果");
        }
        
        // 清理内存
        delete[] h_Rxw;
        delete[] h_Es_IFFT;
        delete[] h_SST_Image;
        delete[] h_scale_weights;
        delete[] h_Enhanced_Rx;
        
        std::cout << "MSST处理完成！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "MSST处理过程中发生错误: " << e.what() << std::endl;
        throw;
    }
}
