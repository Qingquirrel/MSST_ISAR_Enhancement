cmake_policy(SET CMP0146 NEW)
cmake_minimum_required(VERSION 3.10)
project(MSST_ISAR_Enhancement CUDA CXX)

# 设置C++标准和CUDA标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)  # 支持Volta、Turing、Ampere架构

# 寻找CUDA包
find_package(CUDA 12.1 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# 设置CUDA编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70 --use_fast_math")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# 添加CUDA相关库
find_library(CUFFT_LIBRARY cufft HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CUBLAS_LIBRARY cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 添加源文件
set(SOURCES
    src/main.cpp
    src/file_io.cpp
    src/utils.cpp
    src/msst_algorithm.cu
    src/msst_innovations.cu
    src/visualization.cpp
)

# 设置可执行文件输出
add_executable(msst_isar_enhancement ${SOURCES})

# 链接CUDA库
target_link_libraries(msst_isar_enhancement 
    ${CUDA_LIBRARIES}
    ${CUFFT_LIBRARY}
    ${CUBLAS_LIBRARY}
)

# 安装目标
install(TARGETS msst_isar_enhancement DESTINATION bin)
