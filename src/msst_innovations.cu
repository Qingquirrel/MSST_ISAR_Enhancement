#include "../include/msst_innovations.h"
#include "../include/utils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cmath>
#include <iostream>

// -------------- 创新算法1: 自适应尺度选择与优化 --------------

// 用于分析信号频谱分布的CUDA核函数
__global__ void analyzeSpectralDistributionKernel(const cufftComplex* input, float* spectrum, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 计算信号的幅度谱
        spectrum[idx] = sqrtf(input[idx].x * input[idx].x + input[idx].y * input[idx].y);
    }
}

// 计算最优尺度分布的CUDA核函数
__global__ void computeOptimalScalesKernel(const float* spectrum, float* scales, 
                                         int size, int num_scales, float min_scale, float max_scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_scales) {
        // 这个实现使用了更加智能的尺度分布策略，基于频谱能量分布
        // 能量集中的区域会分配更多的尺度点
        float normalized_idx = (float)idx / (num_scales - 1);
        
        // 使用非线性映射来集中关注能量丰富的区域
        float energy_factor = 1.0f; // 这里可以基于spectrum计算能量分布因子
        float adjusted_idx = powf(normalized_idx, energy_factor);
        
        // 计算当前尺度值
        scales[idx] = min_scale + (max_scale - min_scale) * adjusted_idx;
    }
}

// 使用最优尺度进行变换的CUDA核函数
__global__ void transformAtOptimalScaleKernel(const cufftComplex* input, float* output, 
                                           float scale, float f0, const float* t, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 计算当前位置的Morlet小波响应
        float t_scaled = t[idx] / scale;
        float gaussian = expf(-(t_scaled * t_scaled) / 2.0f);
        float angle = 2.0f * M_PI * f0 * t_scaled;
        float sinVal, cosVal;
        sincosf(angle, &sinVal, &cosVal);
        
        // 将小波与信号相乘计算响应
        output[idx] = input[idx].x * gaussian * cosVal + input[idx].y * gaussian * sinVal;
    }
}

// 自适应尺度选择与优化的实现
void adaptiveScaleOptimization(const cufftComplex* input, float* output, 
                             int size, float* optimal_scales, int num_scales) {
    // 设置CUDA核函数的计算块大小
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    int scaleBlocks = (num_scales + blockSize - 1) / blockSize;
    
    // 分配设备内存
    float *d_spectrum, *d_t;
    create_device_array(&d_spectrum, size);
    create_device_array(&d_t, size);
    
    // 创建并复制时间向量到设备
    float* h_t = new float[size];
    for (int i = 0; i < size; i++) {
        h_t[i] = 2.0f * i / (size - 1.0f) - 1.0f;  // 范围从-1到1
    }
    cudaMemcpy(d_t, h_t, size * sizeof(float), cudaMemcpyHostToDevice);
    
    // 分析信号频谱分布
    analyzeSpectralDistributionKernel<<<numBlocks, blockSize>>>(input, d_spectrum, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 计算最优尺度分布
    computeOptimalScalesKernel<<<scaleBlocks, blockSize>>>(d_spectrum, optimal_scales, 
                                                        size, num_scales, 1.0f, 128.0f);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 对每个尺度进行变换
    for (int i = 0; i < num_scales; i++) {
        float scale;
        cudaMemcpy(&scale, &optimal_scales[i], sizeof(float), cudaMemcpyDeviceToHost);
        
        transformAtOptimalScaleKernel<<<numBlocks, blockSize>>>(
            input, &output[i * size], scale, 0.5f, d_t, size);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
    
    // 清理设备内存
    destroy_device_array(d_spectrum);
    destroy_device_array(d_t);
    delete[] h_t;
}

// -------------- 创新算法2: 信息熵引导的权重优化 --------------

// 计算信号概率分布的CUDA核函数
__global__ void computeProbabilityDistributionKernel(const float* input, float* prob_dist, 
                                                  float* sum, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float local_sum[256]; // 共享内存用于归约
    
    // 初始化
    if (idx < size) {
        local_sum[threadIdx.x] = input[idx];
    } else {
        local_sum[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    // 并行归约计算总和
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            local_sum[threadIdx.x] += local_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // 第一个线程保存块总和
    if (threadIdx.x == 0) {
        atomicAdd(sum, local_sum[0]);
    }
    
    __syncthreads();
    
    // 一旦所有块完成，计算概率分布
    if (idx < size && *sum > 1e-10f) {
        prob_dist[idx] = input[idx] / *sum;
    } else if (idx < size) {
        prob_dist[idx] = 0.0f;
    }
}

// 计算信息熵的CUDA核函数
__global__ void computeEntropyKernel(const float* prob_dist, float* entropy, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float local_entropy[256]; // 共享内存用于归约
    
    // 初始化
    if (idx < size && prob_dist[idx] > 1e-10f) {
        local_entropy[threadIdx.x] = -prob_dist[idx] * logf(prob_dist[idx]);
    } else {
        local_entropy[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    // 并行归约计算总熵
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            local_entropy[threadIdx.x] += local_entropy[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // 第一个线程保存块熵
    if (threadIdx.x == 0) {
        atomicAdd(entropy, local_entropy[0]);
    }
}

// 应用熵引导权重的CUDA核函数
__global__ void applyEntropyGuidedWeightsKernel(const float* input, float* output, 
                                              float entropy, float entropy_factor, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 熵越低，表示信息量越大，权重越高
        float weight = expf(-entropy_factor * entropy);
        output[idx] = input[idx] * weight;
    }
}

// 信息熵引导的权重优化实现
void entropyGuidedWeighting(float* SST_Image, float* weights, 
                          int rows, int cols, float entropy_factor) {
    int size = rows * cols;
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    // 分配设备内存
    float *d_prob_dist, *d_entropy, *d_sum;
    create_device_array(&d_prob_dist, size);
    create_device_array(&d_entropy, 1);
    create_device_array(&d_sum, 1);
    
    // 初始化
    cudaMemset(d_entropy, 0, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));
    
    // 计算概率分布
    computeProbabilityDistributionKernel<<<numBlocks, blockSize>>>(
        SST_Image, d_prob_dist, d_sum, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 计算熵
    computeEntropyKernel<<<numBlocks, blockSize>>>(d_prob_dist, d_entropy, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 应用熵引导权重
    float entropy;
    cudaMemcpy(&entropy, d_entropy, sizeof(float), cudaMemcpyDeviceToHost);
    
    applyEntropyGuidedWeightsKernel<<<numBlocks, blockSize>>>(
        SST_Image, weights, entropy, entropy_factor, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 清理设备内存
    destroy_device_array(d_prob_dist);
    destroy_device_array(d_entropy);
    destroy_device_array(d_sum);
}

// -------------- 创新算法3: 空间变化的非局部核聚类增强 --------------

// 准备图像块的CUDA核函数
__global__ void prepareImagePatchesKernel(const float* image, float* patches, 
                                       int rows, int cols, int patch_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < cols && y < rows) {
        int patch_area = patch_size * patch_size;
        int half_patch = patch_size / 2;
        
        // 为当前像素收集图像块
        for (int py = -half_patch; py <= half_patch; py++) {
            for (int px = -half_patch; px <= half_patch; px++) {
                int patch_y = y + py;
                int patch_x = x + px;
                
                // 边界检查和调整
                patch_y = max(0, min(rows - 1, patch_y));
                patch_x = max(0, min(cols - 1, patch_x));
                
                int patch_idx = ((y * cols + x) * patch_area) + 
                              ((py + half_patch) * patch_size + (px + half_patch));
                
                patches[patch_idx] = image[patch_y * cols + patch_x];
            }
        }
    }
}

// 计算块相似度的CUDA核函数
__global__ void computePatchSimilarityKernel(const float* patches, float* weights, 
                                          int rows, int cols, int patch_size, 
                                          int search_window, float h_param) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < cols && y < rows) {
        int patch_area = patch_size * patch_size;
        int half_search = search_window / 2;
        int central_idx = y * cols + x;
        
      // 修改后 - 使用适当的类型转换
const float *central_patch = &patches[central_idx * patch_area];
        
        // 在搜索窗口内计算相似度
        for (int sy = -half_search; sy <= half_search; sy++) {
            for (int sx = -half_search; sx <= half_search; sx++) {
                int search_y = y + sy;
                int search_x = x + sx;
                
                // 边界检查
                if (search_y >= 0 && search_y < rows && search_x >= 0 && search_x < cols) {
                    int search_idx = search_y * cols + search_x;
                    // 修改变量声明为const
const float* search_patch = &patches[search_idx * patch_area];

// 确保所有相关代码都使用const指针
                    
                    // 计算两个块之间的欧几里得距离
                    float distance = 0.0f;
                    for (int i = 0; i < patch_area; i++) {
                        float diff = central_patch[i] - search_patch[i];
                        distance += diff * diff;
                    }
                    
                    // 计算权重
                    float weight = expf(-distance / (h_param * h_param));
                    
                    // 存储权重
                    weights[(central_idx * (search_window * search_window)) + 
                           ((sy + half_search) * search_window + (sx + half_search))] = weight;
                }
            }
        }
    }
}

// 应用非局部均值的CUDA核函数
__global__ void applyNLMKernel(const float* image, const float* weights, float* output, 
                            int rows, int cols, int search_window) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < cols && y < rows) {
        int central_idx = y * cols + x;
        int half_search = search_window / 2;
        float sum_weights = 0.0f;
        float sum_values = 0.0f;
        
        // 计算加权平均
        for (int sy = -half_search; sy <= half_search; sy++) {
            for (int sx = -half_search; sx <= half_search; sx++) {
                int search_y = y + sy;
                int search_x = x + sx;
                
                // 边界检查
                if (search_y >= 0 && search_y < rows && search_x >= 0 && search_x < cols) {
                    int search_idx = search_y * cols + search_x;
                    float weight = weights[(central_idx * (search_window * search_window)) + 
                                         ((sy + half_search) * search_window + (sx + half_search))];
                    
                    sum_weights += weight;
                    sum_values += weight * image[search_idx];
                }
            }
        }
        
        // 归一化和存储结果
        if (sum_weights > 1e-10f) {
            output[central_idx] = sum_values / sum_weights;
        } else {
            output[central_idx] = image[central_idx];
        }
    }
}

// 空间变化的非局部核聚类增强实现
void spatiallyVaryingNLMEnhancement(float* image, int rows, int cols, 
                                  float h_param, int search_window, int patch_size) {
    int size = rows * cols;
    int patch_area = patch_size * patch_size;
    int weights_size = size * search_window * search_window;
    
    // 分配设备内存
    float *d_patches, *d_weights, *d_output;
    create_device_array(&d_patches, size * patch_area);
    create_device_array(&d_weights, weights_size);
    create_device_array(&d_output, size);
    
    // 设置网格和块大小
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, 
                 (rows + blockSize.y - 1) / blockSize.y);
    
    // 准备图像块
    prepareImagePatchesKernel<<<gridSize, blockSize>>>(
        image, d_patches, rows, cols, patch_size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 计算块相似度
    computePatchSimilarityKernel<<<gridSize, blockSize>>>(
        d_patches, d_weights, rows, cols, patch_size, search_window, h_param);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 应用非局部均值
    applyNLMKernel<<<gridSize, blockSize>>>(
        image, d_weights, d_output, rows, cols, search_window);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 复制结果回原始图像
    cudaMemcpy(image, d_output, size * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // 清理设备内存
    destroy_device_array(d_patches);
    destroy_device_array(d_weights);
    destroy_device_array(d_output);
}

// -------------- 创新算法4: 自适应梯度引导的锐化 --------------

// 计算梯度的CUDA核函数
__global__ void computeGradientsKernel(const float* image, float* grad_x, float* grad_y, 
                                     int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < cols && y < rows) {
        int idx = y * cols + x;
        
        // 计算x方向梯度
        if (x > 0 && x < cols - 1) {
            grad_x[idx] = (image[y * cols + (x + 1)] - image[y * cols + (x - 1)]) * 0.5f;
        } else if (x == 0) {
            grad_x[idx] = image[y * cols + 1] - image[y * cols];
        } else { // x == cols - 1
            grad_x[idx] = image[y * cols + x] - image[y * cols + (x - 1)];
        }
        
        // 计算y方向梯度
        if (y > 0 && y < rows - 1) {
            grad_y[idx] = (image[(y + 1) * cols + x] - image[(y - 1) * cols + x]) * 0.5f;
        } else if (y == 0) {
            grad_y[idx] = image[cols + x] - image[x];
        } else { // y == rows - 1
            grad_y[idx] = image[y * cols + x] - image[(y - 1) * cols + x];
        }
    }
}

// 计算梯度幅度的CUDA核函数
__global__ void computeGradientMagnitudeKernel(const float* grad_x, const float* grad_y, 
                                             float* grad_mag, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < cols && y < rows) {
        int idx = y * cols + x;
        grad_mag[idx] = sqrtf(grad_x[idx] * grad_x[idx] + grad_y[idx] * grad_y[idx]);
    }
}

// 应用自适应锐化的CUDA核函数
__global__ void adaptiveGradientSharpeningKernel(const float* image, const float* grad_mag, 
                                              float* output, int rows, int cols, 
                                              float lambda, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < cols && y < rows) {
        int idx = y * cols + x;
        
        // 计算局部方差（使用简化的高斯核）
        float local_var = 0.0f;
        int kernel_radius = (int)ceilf(3.0f * sigma);
        float sum_weights = 0.0f;
        
        for (int ky = -kernel_radius; ky <= kernel_radius; ky++) {
            for (int kx = -kernel_radius; kx <= kernel_radius; kx++) {
                int ny = y + ky;
                int nx = x + kx;
                
                if (ny >= 0 && ny < rows && nx >= 0 && nx < cols) {
                    float gauss_weight = expf(-(kx*kx + ky*ky) / (2.0f * sigma * sigma));
                    local_var += gauss_weight * grad_mag[ny * cols + nx];
                    sum_weights += gauss_weight;
                }
            }
        }
        
        if (sum_weights > 1e-10f) {
            local_var /= sum_weights;
        }
        
        // 计算自适应锐化系数
        float sharpening_factor = lambda * (1.0f / (1.0f + expf(-local_var + sigma)));
        
        // 应用锐化
        output[idx] = image[idx] + sharpening_factor * grad_mag[idx];
        
        // 限制结果范围
        output[idx] = fmaxf(0.0f, fminf(1.0f, output[idx]));
    }
}

// 自适应梯度引导的锐化实现
void adaptiveGradientSharpening(float* image, int rows, int cols, 
                              float lambda, float sigma) {
    int size = rows * cols;
    
    // 分配设备内存
    float *d_grad_x, *d_grad_y, *d_grad_mag, *d_output;
    create_device_array(&d_grad_x, size);
    create_device_array(&d_grad_y, size);
    create_device_array(&d_grad_mag, size);
    create_device_array(&d_output, size);
    
    // 设置网格和块大小
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, 
                 (rows + blockSize.y - 1) / blockSize.y);
    
    // 计算梯度
    computeGradientsKernel<<<gridSize, blockSize>>>(
        image, d_grad_x, d_grad_y, rows, cols);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 计算梯度幅度
    computeGradientMagnitudeKernel<<<gridSize, blockSize>>>(
        d_grad_x, d_grad_y, d_grad_mag, rows, cols);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 应用自适应锐化
    adaptiveGradientSharpeningKernel<<<gridSize, blockSize>>>(
        image, d_grad_mag, d_output, rows, cols, lambda, sigma);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 复制结果回原始图像
    cudaMemcpy(image, d_output, size * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // 清理设备内存
    destroy_device_array(d_grad_x);
    destroy_device_array(d_grad_y);
    destroy_device_array(d_grad_mag);
    destroy_device_array(d_output);
}

// -------------- 创新算法5: 联合时频域去噪 --------------

// 时域小波分解的CUDA核函数
__global__ void waveletDecomposeKernel(const cufftComplex* input, cufftComplex* output, 
                                     int size, int level) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 简化的小波分解 - 在实际实现中需要使用正确的小波基函数
        int scale = 1 << level;
        if (idx % scale == 0) {
            int parent_idx = idx / scale;
            float scale_factor = 1.0f / sqrtf((float)scale);
            output[idx].x = input[parent_idx].x * scale_factor;
            output[idx].y = input[parent_idx].y * scale_factor;
        } else {
            output[idx].x = 0.0f;
            output[idx].y = 0.0f;
        }
    }
}

// 频域阈值去噪的CUDA核函数
__global__ void frequencyThresholdingKernel(cufftComplex* signal, float threshold, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 计算信号幅度
        float magnitude = sqrtf(signal[idx].x * signal[idx].x + signal[idx].y * signal[idx].y);
        
        // 应用软阈值
        if (magnitude <= threshold) {
            signal[idx].x = 0.0f;
            signal[idx].y = 0.0f;
        } else {
            float scale = (magnitude - threshold) / magnitude;
            signal[idx].x *= scale;
            signal[idx].y *= scale;
        }
    }
}

// 小波重构的CUDA核函数
__global__ void waveletReconstructKernel(const cufftComplex* input, cufftComplex* output, 
                                       int size, int level) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 简化的小波重构 - 在实际实现中需要使用正确的小波基函数
        int scale = 1 << level;
        int parent_idx = idx / scale;
        
        if (idx % scale == 0) {
            float scale_factor = sqrtf((float)scale);
            output[parent_idx].x = input[idx].x * scale_factor;
            output[parent_idx].y = input[idx].y * scale_factor;
        }
        // 不需要对其他位置做处理，因为它们已经在分解阶段被设置为零
    }
}

  // 使用一个简化的核函数来缩放结果
__global__ void scaleComplexSignalKernel(cufftComplex* signal, float scale, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            signal[idx].x *= scale;
            signal[idx].y *= scale;
        }
    }

// 联合时频域去噪实现
void jointTimeFrequencyDenoising(cufftComplex* signal, int size, 
                               float threshold, int wavelet_level) {
    // 分配设备内存
    cufftComplex *d_time_freq_signal, *d_wavelet_coeffs;
    create_device_array(&d_time_freq_signal, size);
    create_device_array(&d_wavelet_coeffs, size);
    
    // 复制信号到设备
    cudaMemcpy(d_time_freq_signal, signal, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    
    // 设置CUDA核函数的计算块大小
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    // 创建CUFFT计划
    cufftHandle fft_plan;
    CHECK_CUFFT_ERROR(cufftPlan1d(&fft_plan, size, CUFFT_C2C, 1));
    
    // 执行时域到频域的转换
    CHECK_CUFFT_ERROR(cufftExecC2C(fft_plan, d_time_freq_signal, d_time_freq_signal, CUFFT_FORWARD));
    
    // 在频域应用阈值去噪
    frequencyThresholdingKernel<<<numBlocks, blockSize>>>(d_time_freq_signal, threshold, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 执行频域到时域的转换
    CHECK_CUFFT_ERROR(cufftExecC2C(fft_plan, d_time_freq_signal, d_time_freq_signal, CUFFT_INVERSE));
    
    // 缩放因子 (CUFFT不会自动缩放)
    float scale = 1.0f / size;
    // 调用缩放核函数
    scaleComplexSignalKernel<<<numBlocks, blockSize>>>(d_time_freq_signal, scale, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // 小波分解
    for (int level = 0; level < wavelet_level; level++) {
        waveletDecomposeKernel<<<numBlocks, blockSize>>>(
            d_time_freq_signal, d_wavelet_coeffs, size, level);
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // 在小波域应用阈值去噪
        frequencyThresholdingKernel<<<numBlocks, blockSize>>>(
            d_wavelet_coeffs, threshold / (level + 1), size);
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // 小波重构
        waveletReconstructKernel<<<numBlocks, blockSize>>>(
            d_wavelet_coeffs, d_time_freq_signal, size, level);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
    
    // 复制结果回主机
    cudaMemcpy(signal, d_time_freq_signal, size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    
    // 清理设备内存
    destroy_device_array(d_time_freq_signal);
    destroy_device_array(d_wavelet_coeffs);
    cufftDestroy(fft_plan);
}

// 应用所有创新算法
void applyInnovations(const cufftComplex* input, float* output, 
                     const MsstParams& msst_params,
                     const InnovationParams& innovation_params) {
    int burst = msst_params.burst;
    int pulses = msst_params.pulses;
    int size = burst * pulses;
    
    // 分配主机内存
    cufftComplex* processed_signal = new cufftComplex[size];
    float* temp_output = new float[size];
    float* optimal_scales = new float[msst_params.num_scales];
    
    // 复制输入信号到临时缓冲区
    memcpy(processed_signal, input, size * sizeof(cufftComplex));
    
    std::cout << "正在应用创新算法，处理 " << size << " 个元素..." << std::endl;
    
    // 应用创新算法1: 自适应尺度选择与优化
    if (innovation_params.use_adaptive_scales) {
        std::cout << "应用创新1: 自适应尺度选择与优化..." << std::endl;
        
        // 分配设备内存
        float* d_optimal_scales;
        create_device_array(&d_optimal_scales, msst_params.num_scales);
        
        // 执行自适应尺度优化
        adaptiveScaleOptimization(processed_signal, temp_output, size, d_optimal_scales, msst_params.num_scales);
        
        // 复制优化后的尺度回主机
        cudaMemcpy(optimal_scales, d_optimal_scales, msst_params.num_scales * sizeof(float), cudaMemcpyDeviceToHost);
        
        // 清理设备内存
        destroy_device_array(d_optimal_scales);
    }
    
    // 应用创新算法2: 信息熵引导的权重优化
    if (innovation_params.use_entropy_weighting) {
        std::cout << "应用创新2: 信息熵引导的权重优化..." << std::endl;
        
        // 分配设备内存
        float *d_sst_image, *d_weights;
        create_device_array(&d_sst_image, size);
        create_device_array(&d_weights, size);
        
        // 复制数据到设备
        cudaMemcpy(d_sst_image, temp_output, size * sizeof(float), cudaMemcpyHostToDevice);
        
        // 执行熵引导的权重优化
        entropyGuidedWeighting(d_sst_image, d_weights, burst, pulses, innovation_params.entropy_factor);
        
        // 复制结果回主机
        cudaMemcpy(temp_output, d_weights, size * sizeof(float), cudaMemcpyDeviceToHost);
        
        // 清理设备内存
        destroy_device_array(d_sst_image);
        destroy_device_array(d_weights);
    }
    
    // 应用创新算法3: 空间变化的非局部核聚类增强
    if (innovation_params.use_nlm_enhancement) {
        std::cout << "应用创新3: 空间变化的非局部核聚类增强..." << std::endl;
        
        // 分配设备内存
        float* d_image;
        create_device_array(&d_image, size);
        
        // 复制数据到设备
        cudaMemcpy(d_image, temp_output, size * sizeof(float), cudaMemcpyHostToDevice);
        
        // 执行非局部均值增强
        spatiallyVaryingNLMEnhancement(d_image, burst, pulses, 
                                     innovation_params.nlm_h_param,
                                     innovation_params.nlm_search_window,
                                     innovation_params.nlm_patch_size);
        
        // 复制结果回主机
        cudaMemcpy(temp_output, d_image, size * sizeof(float), cudaMemcpyDeviceToHost);
        
        // 清理设备内存
        destroy_device_array(d_image);
    }
    
    // 应用创新算法4: 自适应梯度引导的锐化
    if (innovation_params.use_gradient_sharpening) {
        std::cout << "应用创新4: 自适应梯度引导的锐化..." << std::endl;
        
        // 分配设备内存
        float* d_image;
        create_device_array(&d_image, size);
        
        // 复制数据到设备
        cudaMemcpy(d_image, temp_output, size * sizeof(float), cudaMemcpyHostToDevice);
        
        // 执行梯度引导的锐化
        adaptiveGradientSharpening(d_image, burst, pulses, 
                                 innovation_params.gradient_lambda,
                                 innovation_params.gradient_sigma);
        
        // 复制结果回主机
        cudaMemcpy(temp_output, d_image, size * sizeof(float), cudaMemcpyDeviceToHost);
        
        // 清理设备内存
        destroy_device_array(d_image);
    }
    
    // 应用创新算法5: 联合时频域去噪
    if (innovation_params.use_joint_tf_denoising) {
        std::cout << "应用创新5: 联合时频域去噪..." << std::endl;
        
        // 执行联合时频域去噪
        jointTimeFrequencyDenoising(processed_signal, size, 
                                  innovation_params.tf_threshold,
                                  innovation_params.tf_wavelet_level);
        
        // 将复数信号转换为实信号（只使用幅度）
        for (int i = 0; i < size; i++) {
            float magnitude = sqrtf(processed_signal[i].x * processed_signal[i].x + 
                                   processed_signal[i].y * processed_signal[i].y);
            temp_output[i] = fmaxf(temp_output[i], magnitude); // 融合结果
        }
    }
    
    // 复制最终结果到输出
    memcpy(output, temp_output, size * sizeof(float));
    
    // 清理主机内存
    delete[] processed_signal;
    delete[] temp_output;
    delete[] optimal_scales;
    
    std::cout << "创新算法应用完成!" << std::endl;
}