#ifndef FILE_IO_H
#define FILE_IO_H

#include <string>
#include <vector>
#include <cufft.h>

// 从二进制文件读取复数数据
void readRowwiseComplexData(const char* filename, std::vector<cufftComplex>& data);

// 从二进制文件读取复数数据到二维矩阵
void readComplexMatrix(const char* filename, cufftComplex* matrix, int rows, int cols);

// 保存复数矩阵到二进制文件
void saveComplexMatrix(const char* filename, const cufftComplex* matrix, int rows, int cols);

// 保存浮点矩阵到二进制文件
void saveFloatMatrix(const char* filename, const float* matrix, int rows, int cols);

// 保存ISAR图像为PNG
bool saveImageAsPNG(const std::string& filename, const float* data, int width, int height, 
                    float min_value = -30.0f, float max_value = 0.0f, 
                    const std::string& colormap = "hot");

#endif // FILE_IO_H