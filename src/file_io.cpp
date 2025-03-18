#include "../include/file_io.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <tuple>
#include <functional>
// 实现从你提供的代码
void readRowwiseComplexData(const char* filename, std::vector<cufftComplex>& data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening data file" << std::endl;
        return;
    }
    
    // 读取实部
    for (int i = 0; i < data.size(); ++i) {
        float realPart;
        file.read(reinterpret_cast<char*>(&realPart), sizeof(float));
        if (!file) {
            std::cerr << "Error reading real part from file" << std::endl;
            return;
        }
        data[i].x = realPart;
    }
    
    // 读取虚部
    for (int i = 0; i < data.size(); ++i) {
        float imagPart;
        file.read(reinterpret_cast<char*>(&imagPart), sizeof(float));
        if (!file) {
            std::cerr << "Error reading imaginary part from file" << std::endl;
            return;
        }
        data[i].y = imagPart;
    }
}

// 读取复数矩阵
void readComplexMatrix(const char* filename, cufftComplex* matrix, int rows, int cols) {
    std::vector<cufftComplex> data(rows * cols);
    readRowwiseComplexData(filename, data);
    
    // 复制到矩阵
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = data[i];
    }
}

// 保存复数矩阵到二进制文件
void saveComplexMatrix(const char* filename, const cufftComplex* matrix, int rows, int cols) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    
    // 写入矩阵维度
    file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    
    // 写入实部
    for (int i = 0; i < rows * cols; ++i) {
        file.write(reinterpret_cast<const char*>(&matrix[i].x), sizeof(float));
    }
    
    // 写入虚部
    for (int i = 0; i < rows * cols; ++i) {
        file.write(reinterpret_cast<const char*>(&matrix[i].y), sizeof(float));
    }
    
    file.close();
}

// 保存浮点矩阵到二进制文件
void saveFloatMatrix(const char* filename, const float* matrix, int rows, int cols) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    
    // 写入矩阵维度
    file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    
    // 写入数据
    file.write(reinterpret_cast<const char*>(matrix), rows * cols * sizeof(float));
    
    file.close();
}

// 保存ISAR图像为简单的PPM格式（由于环境限制，不使用复杂的图像库）
bool saveImageAsPNG(const std::string& filename, const float* data, int width, int height, 
                  float min_value, float max_value, const std::string& colormap) {
    // 创建PPM格式的图像文件(P6格式)
    std::string ppm_filename = filename + ".ppm";
    std::ofstream file(ppm_filename, std::ios::binary);
    
    if (!file) {
        std::cerr << "Error: Could not open file " << ppm_filename << " for writing." << std::endl;
        return false;
    }
    
    // PPM头
    file << "P6\n";
    file << width << " " << height << "\n";
    file << "255\n";
    
    // 创建颜色映射函数
    std::function<std::tuple<unsigned char, unsigned char, unsigned char>(float)> selected_colormap;
    
    // 热力图颜色映射
    auto hot_colormap = [](float value) -> std::tuple<unsigned char, unsigned char, unsigned char> {
        unsigned char r, g, b;
        
        if (value < 1.0f/3.0f) {
            r = static_cast<unsigned char>(255 * value * 3);
            g = 0;
            b = 0;
        } else if (value < 2.0f/3.0f) {
            r = 255;
            g = static_cast<unsigned char>(255 * (value - 1.0f/3.0f) * 3);
            b = 0;
        } else {
            r = 255;
            g = 255;
            b = static_cast<unsigned char>(255 * (value - 2.0f/3.0f) * 3);
        }
        
        return std::make_tuple(r, g, b);
    };
    
    // 灰度色谱图
    auto gray_colormap = [](float value) -> std::tuple<unsigned char, unsigned char, unsigned char> {
        unsigned char intensity = static_cast<unsigned char>(255 * value);
        return std::make_tuple(intensity, intensity, intensity);
    };
    
    // Jet 颜色映射
    auto jet_colormap = [](float value) -> std::tuple<unsigned char, unsigned char, unsigned char> {
        unsigned char r, g, b;
        
        if (value < 0.125f) {
            r = 0;
            g = 0;
            b = static_cast<unsigned char>(255 * (0.5f + value * 4));
        } else if (value < 0.375f) {
            r = 0;
            g = static_cast<unsigned char>(255 * ((value - 0.125f) * 4));
            b = 255;
        } else if (value < 0.625f) {
            r = static_cast<unsigned char>(255 * ((value - 0.375f) * 4));
            g = 255;
            b = static_cast<unsigned char>(255 * (1.0f - (value - 0.375f) * 4));
        } else if (value < 0.875f) {
            r = 255;
            g = static_cast<unsigned char>(255 * (1.0f - (value - 0.625f) * 4));
            b = 0;
        } else {
            r = static_cast<unsigned char>(255 * (1.0f - (value - 0.875f) * 4));
            g = 0;
            b = 0;
        }
        
        return std::make_tuple(r, g, b);
    };
    
    // 选择颜色映射
    if (colormap == "gray") {
        selected_colormap = gray_colormap;
    } else if (colormap == "jet") {
        selected_colormap = jet_colormap;
    } else {
        selected_colormap = hot_colormap; // 默认为热力图
    }
    
    // 写入像素数据
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 获取数据点（假设数据是dB单位）
            float value = data[y * width + x];
            
            // 归一化到[0,1]范围
            float normalized_value = (value - min_value) / (max_value - min_value);
            normalized_value = std::max(0.0f, std::min(1.0f, normalized_value));
            
            // 应用颜色映射
            unsigned char r, g, b;
            std::tie(r, g, b) = selected_colormap(normalized_value);
            
            // 写入RGB值
            file.put(r);
            file.put(g);
            file.put(b);
        }
    }
    
    file.close();
    std::cout << "Image saved as " << ppm_filename << std::endl;
    return true;
}