#ifndef MSST_INNOVATIONS_H
#define MSST_INNOVATIONS_H

#include <cufft.h>
#include "msst_algorithm.h"

// 创新算法1: 自适应尺度选择与优化
void adaptiveScaleOptimization(const cufftComplex* input, float* output, 
                               int size, float* optimal_scales, int num_scales);

// 创新算法2: 信息熵引导的权重优化
void entropyGuidedWeighting(float* SST_Image, float* weights, 
                            int rows, int cols, float entropy_factor);

// 创新算法3: 空间变化的非局部核聚类增强
void spatiallyVaryingNLMEnhancement(float* image, int rows, int cols, 
                                   float h_param, int search_window, int patch_size);

// 创新算法4: 自适应梯度引导的锐化
void adaptiveGradientSharpening(float* image, int rows, int cols, 
                               float lambda, float sigma);

// 创新算法5: 联合时频域去噪
void jointTimeFrequencyDenoising(cufftComplex* signal, int size, 
                                 float threshold, int wavelet_level);

// 创新算法参数结构体
struct InnovationParams {
    bool use_adaptive_scales;       // 是否使用自适应尺度选择
    bool use_entropy_weighting;     // 是否使用熵引导权重
    bool use_nlm_enhancement;       // 是否使用非局部核聚类增强
    bool use_gradient_sharpening;   // 是否使用梯度引导锐化
    bool use_joint_tf_denoising;    // 是否使用联合时频域去噪
    
    float entropy_factor;          // 熵权重因子
    float nlm_h_param;             // 非局部均值h参数
    int nlm_search_window;         // 非局部均值搜索窗口
    int nlm_patch_size;            // 非局部均值块大小
    float gradient_lambda;         // 梯度锐化lambda
    float gradient_sigma;          // 梯度锐化sigma
    float tf_threshold;            // 时频域阈值
    int tf_wavelet_level;          // 时频域小波级别
};

// 应用所有创新算法
void applyInnovations(const cufftComplex* input, float* output, 
                      const MsstParams& msst_params,
                      const InnovationParams& innovation_params);

#endif // MSST_INNOVATIONS_H