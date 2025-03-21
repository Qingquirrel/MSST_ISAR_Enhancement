#ifndef MSST_ALGORITHM_H
#define MSST_ALGORITHM_H

#include <cufft.h>
#include <vector>
#include <cfloat>  // 添加这个头文件来解决 FLT_MAX 未定义的问题

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
// 修改参数类型以匹配 msst_algorithm.cu 中的实现
void processMSST(cufftComplex* h_Rx, float* h_Enhanced_ISAR, const MsstParams& params);

// 窗口函数应用
// 修改参数名称以匹配实现
void applyKaiserWindow(cufftComplex* h_Rx, cufftComplex* h_Rxw, int burst, int pulses, float beta);

// 逆FFT获取距离分布
// 修改参数名称以匹配实现
void performIFFT(cufftComplex* h_Rxw, cufftComplex* h_Es_IFFT, int burst, int pulses);

// 多尺度小波变换
// 修改参数名称和类型以匹配实现
void calculateMorletWavelet(cufftComplex* h_Es_IFFT, float* h_SST_Image, 
                           float* h_scale_weights, const MsstParams& params);

// 融合和增强SST图像
// 修改参数名称和类型以匹配实现
void fuseAndEnhanceSST(float* h_SST_Image, float* h_scale_weights, 
                      float* h_Enhanced_Rx, const MsstParams& params);

// 生成最终ISAR图像
// 修改参数名称以匹配实现
void generateISARImage(float* h_Enhanced_Rx, float* h_Enhanced_ISAR, const MsstParams& params);

#endif // MSST_ALGORITHM_H