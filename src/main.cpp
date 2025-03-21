#include <iostream>
#include <string>
#include <vector>
#include <cufft.h>
#include <cuda_runtime.h>

#include "../include/utils.h"
#include "../include/file_io.h"
#include "../include/msst_algorithm.h"
#include "../include/msst_innovations.h"
#include "../include/visualization.h"

int main(int argc, char* argv[]) {
    // 打印CUDA设备信息
    print_cuda_device_info();
    
    // 解析命令行参数
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file> [output_file]" << std::endl;
        return 1;
    }
    
    const char* input_file = argv[1];
    std::string output_file = (argc > 2) ? argv[2] : "enhanced_isar.bin";
    std::string image_file = "enhanced_isar_image.ppm";
    
    try {
        // 创建计时器并启动
        Timer timer;
        timer.start();
        
        // 设置MSST参数 (根据你的MATLAB代码)
        MsstParams msst_params;
        msst_params.burst = 500;         // 行数 (调整为实际值)
        msst_params.pulses = 250;        // 列数 (调整为实际值)
        msst_params.beta = 5.0f;         // Kaiser窗口参数
        msst_params.f0 = 0.5f;           // Morlet小波中心频率
        msst_params.num_scales = 512;    // 小波尺度数量
        msst_params.min_scale = 1.0f;    // 最小尺度
        msst_params.max_scale = 128.0f;  // 最大尺度
        msst_params.gain_factor = 2.0f;  // 增益因子
        msst_params.non_linear_power = 0.8f; // 非线性拉伸指数
        msst_params.dr = 0.025f;         // 距离分辨率
        
        // 设置创新算法参数
        InnovationParams innovation_params;
        innovation_params.use_adaptive_scales = true;
        innovation_params.use_entropy_weighting = true;
        innovation_params.use_nlm_enhancement = true;
        innovation_params.use_gradient_sharpening = true;
        innovation_params.use_joint_tf_denoising = true;
        
        innovation_params.entropy_factor = 1.0f;
        innovation_params.nlm_h_param = 10.0f;
        innovation_params.nlm_search_window = 7;
        innovation_params.nlm_patch_size = 3;
        innovation_params.gradient_lambda = 0.1f;
        innovation_params.gradient_sigma = 1.0f;
        innovation_params.tf_threshold = 0.05f;
        innovation_params.tf_wavelet_level = 3;
        
        std::cout << "Initializing MSST processing with dimensions: " 
                  << msst_params.burst << " x " << msst_params.pulses << std::endl;
        
        // 分配主机内存
        cufftComplex* h_Rx = new cufftComplex[msst_params.burst * msst_params.pulses];
        float* h_Enhanced_ISAR = new float[msst_params.burst * msst_params.pulses];
        
        // 读取输入数据
        std::cout << "Reading input data from: " << input_file << std::endl;
        readComplexMatrix(input_file, h_Rx, msst_params.burst, msst_params.pulses);
        
        // 处理MSST算法
        std::cout << "Running MSST algorithm..." << std::endl;
        processMSST(h_Rx, h_Enhanced_ISAR, msst_params);
        
        // 应用创新算法增强
       // std::cout << "Applying innovation algorithms..." << std::endl;
        //applyInnovations(h_Rx, h_Enhanced_ISAR, msst_params, innovation_params);
        
        // 保存结果
        std::cout << "Saving enhanced ISAR data to: " << output_file << std::endl;
        saveFloatMatrix(output_file.c_str(), h_Enhanced_ISAR, msst_params.burst, msst_params.pulses);
        
        // 保存可视化图像
        std::cout << "Creating visualization image: " << image_file << std::endl;
        saveImageAsPNG(image_file, h_Enhanced_ISAR, msst_params.pulses, msst_params.burst, -30.0f, 0.0f, "hot");
        
        // 停止计时
        timer.stop();
        std::cout << "Processing completed in " << timer.elapsed_seconds() << " seconds." << std::endl;
        
        // 释放主机内存
        delete[] h_Rx;
        delete[] h_Enhanced_ISAR;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}