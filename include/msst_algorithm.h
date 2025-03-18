#ifndef MSST_ALGORITHM_H
#define MSST_ALGORITHM_H

#include <cufft.h>
#include <vector>

// MSST参数结构体
struct MsstParams {
    int burst;              // 目标方向大小
    int pulses;             // 脉冲方向大小
    float beta;             // Kaiser窗口参数
    float f0;               // Morlet小波中心频率
    int num_scales;         // 小波尺度数量
    float min_scale;        // 最小尺度
    float max_scale;        // 最大尺度
    float gain_factor;      // 增益因子
    float non_linear_power; // 非线性拉伸指数
    float dr;               // 距离分辨率
};

// MSST处理函数 - 主函数
void processMSST(const cufftComplex* Rx, float* Enhanced_ISAR, const MsstParams& params);

// 窗口函数应用
void applyKaiserWindow(const cufftComplex* Rx, cufftComplex* Rxw, int burst, int pulses, float beta);

// 逆FFT获取距离分布
void performIFFT(const cufftComplex* Rxw, cufftComplex* Es_IFFT, int burst, int pulses);

// 多尺度小波变换
void calculateMorletWavelet(const cufftComplex* Es_IFFT, float* SST_Image, 
                           float* scale_weights, const MsstParams& params);

// 融合和增强SST图像
void fuseAndEnhanceSST(const float* SST_Image, const float* scale_weights, 
                      float* Enhanced_Rx, const MsstParams& params);

// 生成最终ISAR图像
void generateISARImage(const float* Enhanced_Rx, float* Enhanced_ISAR, const MsstParams& params);

#endif // MSST_ALGORITHM_H