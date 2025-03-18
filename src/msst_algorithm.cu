#include "../include/msst_algorithm.h"
#include "../include/utils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cmath>
#include <iostream>

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
        window[row * pulses + col] = win_burst[row] * win_pulses[col];
    }
}

// CUDA核心函数：复数乘法
__global__ void complexMultiplyKernel(const cufftComplex* a, const cufftComplex* b, 
                                    cufftComplex* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float real = a[idx].x * b[idx].x - a[idx].y * b[idx].y;
        float imag = a[idx].x * b[idx].y + a[idx].y * b[idx].x;
        result[idx].x = real;
        result[idx].y = imag;
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

// CUDA核心函数：复数向量的绝对值
__global__ void complexAbsKernel(const cufftComplex* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = sqrtf(input[idx].x * input[idx].x + input[idx].y * input[idx].y);
    }
}

// CUDA核心函数：应用权重
__global__ void applyWeightsKernel(const float* input, float* output, const float* weights, 
                                int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * weights[idx];
    }
}

// CUDA核心函数：应用熵优化
__global__ void applyEntropyOptimizationKernel(const float* input, float* output, 
                                            float entropy, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * expf(-entropy);
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
void applyKaiserWindow(const cufftComplex* Rx, cufftComplex* Rxw, int burst, int pulses, float beta) {
    int size = burst * pulses;
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    // 在主机上生成凯撒窗口
    float* h_win_burst = new float[burst];
    float* h_win_pulses = new float[pulses];
    
    generate_kaiser_window(h_win_burst, burst, beta);
    generate_kaiser_window(h_win_pulses, pulses, beta);
    
    // 将窗口复制到设备
    float *d_win_burst, *d_win_pulses, *d_window;
    create_device_array(&d_win_burst, burst);
    create_device_array(&d_win_pulses, pulses);
    create_device_array(&d_window, size);
    
    cudaMemcpy(d_win_burst, h_win_burst, burst * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_win_pulses, h_win_pulses, pulses * sizeof(float), cudaMemcpyHostToDevice);
    
    // 生成2D窗口
    dim3 blockDim(16, 16);
    dim3 gridDim((pulses + blockDim.x - 1) / blockDim.x, (burst + blockDim.y - 1) / blockDim.y);
    
    generate2DKaiserWindowKernel<<<gridDim, blockDim>>>(d_window, d_win_burst, d_win_pulses, burst, pulses);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 应用窗口
    applyKaiserWindowKernel<<<numBlocks, blockSize>>>(Rx, Rxw, d_window, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 清理
    destroy_device_array(d_win_burst);
    destroy_device_array(d_win_pulses);
    destroy_device_array(d_window);
    delete[] h_win_burst;
    delete[] h_win_pulses;
}

// 执行逆FFT
void performIFFT(const cufftComplex* Rxw, cufftComplex* Es_IFFT, int burst, int pulses) {
    int size = burst * pulses;
    
    // 创建CUFFT计划
    cufftHandle plan;
    CHECK_CUFFT_ERROR(cufftPlan1d(&plan, size, CUFFT_C2C, 1));
    
    // 执行逆FFT
    CHECK_CUFFT_ERROR(cufftExecC2C(plan, (cufftComplex*)Rxw, Es_IFFT, CUFFT_INVERSE));
    
    // 销毁计划
    cufftDestroy(plan);
    
    // 标准化
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    // 注意：CUFFT不会自动归一化结果，如需要可以手动归一化
}

// 计算Morlet小波变换
void calculateMorletWavelet(const cufftComplex* Es_IFFT, float* SST_Image, 
                           float* scale_weights, const MsstParams& params) {
    int burst = params.burst;
    int pulses = params.pulses;
    int size = burst * pulses;
    int num_scales = params.num_scales;
    
    // 分配设备内存
    cufftComplex *d_wavelet, *d_convolved;
    float *d_t, *d_abs_convolved, *d_weights, *d_weighted, *d_entropy;
    
    create_device_array(&d_wavelet, size);
    create_device_array(&d_convolved, size);
    create_device_array(&d_t, size);
    create_device_array(&d_abs_convolved, size);
    create_device_array(&d_weights, size);
    create_device_array(&d_weighted, size);
    create_device_array(&d_entropy, 1);
    
    // 创建时间向量
    float* h_t = new float[size];
    for (int i = 0; i < size; i++) {
        h_t[i] = 2.0f * i / (size - 1.0f) - 1.0f;  // 范围从-1到1
    }
    cudaMemcpy(d_t, h_t, size * sizeof(float), cudaMemcpyHostToDevice);
    
    // 创建CUFFT计划 (用于卷积)
    cufftHandle plan;
    CHECK_CUFFT_ERROR(cufftPlan1d(&plan, size, CUFFT_C2C, 1));
    
    // 设置内核参数
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    // 为每个尺度生成Morlet小波并计算变换
    for (int s = 0; s < num_scales; s++) {
        // 计算当前尺度
        float scale = params.min_scale + (params.max_scale - params.min_scale) * s / (num_scales - 1);
        
        // 生成Morlet小波
        morletWaveletKernel<<<numBlocks, blockSize>>>(d_wavelet, scale, params.f0, d_t, size);
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // 计算FFT(wavelet)
        CHECK_CUFFT_ERROR(cufftExecC2C(plan, d_wavelet, d_wavelet, CUFFT_FORWARD));
        
        // 频域乘法实现卷积
        complexMultiplyKernel<<<numBlocks, blockSize>>>(d_wavelet, (cufftComplex*)Es_IFFT, d_convolved, size);
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // 逆FFT
        CHECK_CUFFT_ERROR(cufftExecC2C(plan, d_convolved, d_convolved, CUFFT_INVERSE));
        
        // 计算绝对值
        complexAbsKernel<<<numBlocks, blockSize>>>(d_convolved, d_abs_convolved, size);
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // 计算权重 (使用高斯权重)
        applyWeightsKernel<<<numBlocks, blockSize>>>(d_abs_convolved, d_weighted, d_abs_convolved, size);
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // 计算当前尺度的权重 (基于方差的倒数)
        // 这里简化为固定权重，实际应根据信号特性动态计算
        scale_weights[s] = 1.0f / (1.0f + s * 0.01f);
        
        // 复制结果到SST图像
        cudaMemcpy(&SST_Image[s * size], d_weighted, size * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    // 规范化尺度权重
    float sum_weights = 0.0f;
    for (int s = 0; s < num_scales; s++) {
        sum_weights += scale_weights[s];
    }
    for (int s = 0; s < num_scales; s++) {
        scale_weights[s] /= sum_weights;
    }
    
    // 清理
    cufftDestroy(plan);
    destroy_device_array(d_wavelet);
    destroy_device_array(d_convolved);
    destroy_device_array(d_t);
    destroy_device_array(d_abs_convolved);
    destroy_device_array(d_weights);
    destroy_device_array(d_weighted);
    destroy_device_array(d_entropy);
    delete[] h_t;
}

// 融合和增强SST图像
// 融合和增强SST图像
void fuseAndEnhanceSST(const float* SST_Image, const float* scale_weights, 
                      float* Enhanced_Rx, const MsstParams& params) {
    int burst = params.burst;
    int pulses = params.pulses;
    int size = burst * pulses;
    int num_scales = params.num_scales;
    
    // 分配设备内存
    float *d_sst_image, *d_scale_weights, *d_fused_sst;
    create_device_array(&d_sst_image, num_scales * size);
    create_device_array(&d_scale_weights, num_scales);
    create_device_array(&d_fused_sst, size);
    
    // 复制数据到设备
    cudaMemcpy(d_sst_image, SST_Image, num_scales * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_weights, scale_weights, num_scales * sizeof(float), cudaMemcpyHostToDevice);
    
    // 设置内核参数
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    // 融合多尺度结果
    fuseScalesKernel<<<numBlocks, blockSize>>>(d_sst_image, d_fused_sst, d_scale_weights, num_scales, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 非线性拉伸和增益
    nonLinearStretchAndGainKernel<<<numBlocks, blockSize>>>(d_fused_sst, params.non_linear_power, params.gain_factor, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 将结果复制回主机
    cudaMemcpy(Enhanced_Rx, d_fused_sst, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 清理设备内存
    destroy_device_array(d_sst_image);
    destroy_device_array(d_scale_weights);
    destroy_device_array(d_fused_sst);
}
// CUDA核函数：执行FFT移位
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

// 生成最终ISAR图像
void generateISARImage(const float* Enhanced_Rx, float* Enhanced_ISAR, const MsstParams& params) {
    int burst = params.burst;
    int pulses = params.pulses;
    int size = burst * pulses;
    
    // 分配设备内存
    cufftComplex *d_input, *d_output;
    float *d_magnitude;
    
    allocate_complex_vector(&d_input, size);
    allocate_complex_vector(&d_output, size);
    create_device_array(&d_magnitude, size);
    
    // 将实数数据转换为复数形式（虚部为零）
    cufftComplex* h_input = new cufftComplex[size];
    for (int i = 0; i < size; i++) {
        h_input[i].x = Enhanced_Rx[i];
        h_input[i].y = 0.0f;
    }
    
    // 复制数据到设备
    copy_complex_vector_host_to_device(d_input, h_input, size);
    
    // 创建2D FFT计划
    cufftHandle plan;
    int dims[2] = {burst, pulses};
    CHECK_CUFFT_ERROR(cufftPlanMany(&plan, 2, dims, 
                                  NULL, 1, size,  // 输入数据布局
                                  NULL, 1, size,  // 输出数据布局
                                  CUFFT_C2C, 1)); // 复数到复数
    
    // 执行FFT
    CHECK_CUFFT_ERROR(cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD));
    
// 执行FFT移位并计算幅度
dim3 blockDim2D(16, 16);
dim3 gridDim2D((pulses + blockDim2D.x - 1) / blockDim2D.x, 
             (burst + blockDim2D.y - 1) / blockDim2D.y);
    
    fftshiftKernel<<<gridDim2D, blockDim2D>>>(d_output, d_magnitude, burst, pulses);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 将结果复制回主机
    cudaMemcpy(Enhanced_ISAR, d_magnitude, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 归一化ISAR图像
    float max_val = 0.0f;
    for (int i = 0; i < size; i++) {
        // 修改后 - 使用适当的类型
    max_val = std::max<float>(max_val, Enhanced_ISAR[i]);
    }
    
    if (max_val > 1e-10f) {
        for (int i = 0; i < size; i++) {
            Enhanced_ISAR[i] /= max_val;
        }
    }
    
    // 清理内存
    delete[] h_input;
    free_complex_vector(d_input);
    free_complex_vector(d_output);
    destroy_device_array(d_magnitude);
    cufftDestroy(plan);
}
// MSST处理函数 - 主函数
void processMSST(const cufftComplex* Rx, float* Enhanced_ISAR, const MsstParams& params) {
    int burst = params.burst;
    int pulses = params.pulses;
    int size = burst * pulses;
    
    std::cout << "开始MSST处理，矩阵大小: " << burst << "x" << pulses << std::endl;
    
    // 分配设备内存
    cufftComplex *d_Rx, *d_Rxw, *d_Es_IFFT;
    float *d_SST_Image, *d_scale_weights, *d_Enhanced_Rx, *d_Enhanced_ISAR;
    
    // 为设备分配内存
    allocate_complex_vector(&d_Rx, size);
    allocate_complex_vector(&d_Rxw, size);
    allocate_complex_vector(&d_Es_IFFT, size);
    
    create_device_array(&d_SST_Image, params.num_scales * size);
    create_device_array(&d_scale_weights, params.num_scales);
    create_device_array(&d_Enhanced_Rx, size);
    create_device_array(&d_Enhanced_ISAR, size);
    
    // 分配主机内存
    float* h_SST_Image = new float[params.num_scales * size];
    float* h_scale_weights = new float[params.num_scales];
    float* h_Enhanced_Rx = new float[size];
    
    // 复制输入数据到设备
    copy_complex_vector_host_to_device(d_Rx, Rx, size);
    
    // 步骤1: 应用Kaiser窗口
    std::cout << "步骤1: 应用Kaiser窗口..." << std::endl;
    applyKaiserWindow(d_Rx, d_Rxw, burst, pulses, params.beta);
    
    // 步骤2: 执行逆FFT获取距离分布
    std::cout << "步骤2: 执行逆FFT获取距离分布..." << std::endl;
    performIFFT(d_Rxw, d_Es_IFFT, burst, pulses);
    
    // 步骤3: 多尺度小波变换
    std::cout << "步骤3: 多尺度小波变换分析..." << std::endl;
    calculateMorletWavelet(d_Es_IFFT, h_SST_Image, h_scale_weights, params);
    
    // 复制SST图像和权重到设备
    cudaMemcpy(d_SST_Image, h_SST_Image, params.num_scales * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_weights, h_scale_weights, params.num_scales * sizeof(float), cudaMemcpyHostToDevice);
    
    // 步骤4: 融合和增强SST图像
    std::cout << "步骤4: 融合和增强SST图像..." << std::endl;
    fuseAndEnhanceSST(d_SST_Image, d_scale_weights, h_Enhanced_Rx, params);
    
    // 复制增强后的数据到设备
    cudaMemcpy(d_Enhanced_Rx, h_Enhanced_Rx, size * sizeof(float), cudaMemcpyHostToDevice);
    
    // 步骤5: 生成最终ISAR图像
    std::cout << "步骤5: 生成最终ISAR图像..." << std::endl;
    generateISARImage(h_Enhanced_Rx, Enhanced_ISAR, params);
    
    // 清理内存
    free_complex_vector(d_Rx);
    free_complex_vector(d_Rxw);
    free_complex_vector(d_Es_IFFT);
    
    destroy_device_array(d_SST_Image);
    destroy_device_array(d_scale_weights);
    destroy_device_array(d_Enhanced_Rx);
    destroy_device_array(d_Enhanced_ISAR);
    
    delete[] h_SST_Image;
    delete[] h_scale_weights;
    delete[] h_Enhanced_Rx;
    
    std::cout << "MSST处理完成！" << std::endl;
}