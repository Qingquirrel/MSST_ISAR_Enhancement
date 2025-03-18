#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <string>

// 将ISAR图像数据保存为可视化格式
void saveISARVisualization(const float* data, int rows, int cols, 
                         const std::string& filename, float min_val, float max_val);

// 生成ISAR热力图的颜色映射
void generateColorMap(unsigned char* colormap, int size, const std::string& map_name);

// 将dB尺度的图像归一化到0-255范围
void normalizeDbImage(const float* db_image, unsigned char* normalized_image, 
                    int size, float min_db, float max_db);

// 生成X轴和Y轴标签数据
void generateAxisLabels(float* x_labels, float* y_labels, int x_size, int y_size, 
                      float x_min, float x_max, float y_min, float y_max);

#endif // VISUALIZATION_H