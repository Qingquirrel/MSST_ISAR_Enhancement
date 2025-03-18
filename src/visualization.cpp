#include "../include/visualization.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>

// 将ISAR图像数据保存为可视化格式
void saveISARVisualization(const float* data, int rows, int cols, 
                         const std::string& filename, float min_val, float max_val) {
    // 创建dB尺度的图像
    std::vector<float> db_image(rows * cols);
    
    // 转换为dB尺度
    for (int i = 0; i < rows * cols; i++) {
        if (data[i] > 1e-10f) {
            db_image[i] = 20.0f * log10f(data[i]);
        } else {
            db_image[i] = min_val;
        }
    }
    
    // 归一化到0-255范围用于可视化
    std::vector<unsigned char> normalized_image(rows * cols);
    normalizeDbImage(db_image.data(), normalized_image.data(), rows * cols, min_val, max_val);
    
    // 生成颜色映射
    unsigned char colormap[256 * 3]; // RGB颜色映射
    generateColorMap(colormap, 256, "hot");
    
    // 创建PPM格式的图像文件(P6格式)
    std::string ppm_filename = filename + ".ppm";
    std::ofstream file(ppm_filename, std::ios::binary);
    
    if (!file) {
        std::cerr << "错误：无法打开文件 " << ppm_filename << " 进行写入。" << std::endl;
        return;
    }
    
    // PPM头部
    file << "P6\n";
    file << cols << " " << rows << "\n";
    file << "255\n";
    
    // 应用颜色映射并写入像素数据
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            int idx = y * cols + x;
            int color_idx = normalized_image[idx] * 3;
            
            // 写入RGB值
            file.put(colormap[color_idx]);
            file.put(colormap[color_idx + 1]);
            file.put(colormap[color_idx + 2]);
        }
    }
    
    file.close();
    std::cout << "图像已保存为 " << ppm_filename << std::endl;
    
    // 生成附加的元数据文件，包含坐标轴信息
    std::string metadata_filename = filename + "_metadata.txt";
    std::ofstream metadata_file(metadata_filename);
    
    if (metadata_file) {
        // 生成坐标轴标签
        float* x_labels = new float[cols];
        float* y_labels = new float[rows];
        generateAxisLabels(x_labels, y_labels, cols, rows, -0.5f, 0.5f, -0.5f, 0.5f);
        
        // 写入元数据
        metadata_file << "# ISAR图像元数据\n";
        metadata_file << "行数: " << rows << "\n";
        metadata_file << "列数: " << cols << "\n";
        metadata_file << "dB范围: [" << min_val << ", " << max_val << "]\n\n";
        
        // 写入X轴标签
        metadata_file << "# X轴标签 (距离, 米)\n";
        for (int i = 0; i < cols; i++) {
            metadata_file << i << ": " << x_labels[i] << "\n";
        }
        
        // 写入Y轴标签
        metadata_file << "\n# Y轴标签 (多普勒指数)\n";
        for (int i = 0; i < rows; i++) {
            metadata_file << i << ": " << y_labels[i] << "\n";
        }
        
        metadata_file.close();
        delete[] x_labels;
        delete[] y_labels;
        
        std::cout << "元数据已保存为 " << metadata_filename << std::endl;
    }
}

// 生成ISAR热力图的颜色映射
void generateColorMap(unsigned char* colormap, int size, const std::string& map_name) {
    if (map_name == "hot") {
        // 热力图颜色映射：黑->红->黄->白
        for (int i = 0; i < size; i++) {
            float normalized = (float)i / (size - 1);
            
            if (normalized < 1.0f/3.0f) {
                // 黑->红
                colormap[i*3] = (unsigned char)(255 * normalized * 3);
                colormap[i*3+1] = 0;
                colormap[i*3+2] = 0;
            } else if (normalized < 2.0f/3.0f) {
                // 红->黄
                colormap[i*3] = 255;
                colormap[i*3+1] = (unsigned char)(255 * (normalized - 1.0f/3.0f) * 3);
                colormap[i*3+2] = 0;
            } else {
                // 黄->白
                colormap[i*3] = 255;
                colormap[i*3+1] = 255;
                colormap[i*3+2] = (unsigned char)(255 * (normalized - 2.0f/3.0f) * 3);
            }
        }
    } else if (map_name == "jet") {
        // Jet颜色映射：蓝->青->绿->黄->红
        for (int i = 0; i < size; i++) {
            float normalized = (float)i / (size - 1);
            
            if (normalized < 0.125f) {
                colormap[i*3] = 0;
                colormap[i*3+1] = 0;
                colormap[i*3+2] = (unsigned char)(255 * (0.5f + normalized * 4));
            } else if (normalized < 0.375f) {
                colormap[i*3] = 0;
                colormap[i*3+1] = (unsigned char)(255 * ((normalized - 0.125f) * 4));
                colormap[i*3+2] = 255;
            } else if (normalized < 0.625f) {
                colormap[i*3] = (unsigned char)(255 * ((normalized - 0.375f) * 4));
                colormap[i*3+1] = 255;
                colormap[i*3+2] = (unsigned char)(255 * (1.0f - (normalized - 0.375f) * 4));
            } else if (normalized < 0.875f) {
                colormap[i*3] = 255;
                colormap[i*3+1] = (unsigned char)(255 * (1.0f - (normalized - 0.625f) * 4));
                colormap[i*3+2] = 0;
            } else {
                colormap[i*3] = (unsigned char)(255 * (1.0f - (normalized - 0.875f) * 4));
                colormap[i*3+1] = 0;
                colormap[i*3+2] = 0;
            }
        }
    } else if (map_name == "gray") {
        // 灰度颜色映射
        for (int i = 0; i < size; i++) {
            unsigned char intensity = (unsigned char)i;
            colormap[i*3] = intensity;
            colormap[i*3+1] = intensity;
            colormap[i*3+2] = intensity;
        }
    } else {
        // 默认为热力图
        generateColorMap(colormap, size, "hot");
    }
}

// 将dB尺度的图像归一化到0-255范围
void normalizeDbImage(const float* db_image, unsigned char* normalized_image, 
                    int size, float min_db, float max_db) {
    for (int i = 0; i < size; i++) {
        // 限制到指定范围
        float value = std::max(min_db, std::min(max_db, db_image[i]));
        
        // 归一化到[0,1]范围
        float normalized_value = (value - min_db) / (max_db - min_db);
        
        // 转换到0-255范围
        normalized_image[i] = (unsigned char)(normalized_value * 255);
    }
}

// 生成X轴和Y轴标签数据
void generateAxisLabels(float* x_labels, float* y_labels, int x_size, int y_size, 
                      float x_min, float x_max, float y_min, float y_max) {
    // 生成X轴标签（距离）
    for (int i = 0; i < x_size; i++) {
        x_labels[i] = x_min + (x_max - x_min) * i / (x_size - 1);
    }
    
    // 生成Y轴标签（多普勒指数）
    for (int i = 0; i < y_size; i++) {
        y_labels[i] = y_min + (y_max - y_min) * i / (y_size - 1);
    }
}