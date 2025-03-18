cmake_minimum_required(VERSION 3.10)
project(MSST_ISAR_Enhancement CUDA CXX)

# 设置C++标准和CUDA标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# 查找CUDA包
find_package(CUDA 12.1 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# 直接指定支持的GPU架构，确保包含Tesla P100的计算能力6.0
# 简化架构列表以避免检测问题
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -gencode arch=compute_50,code=sm_50
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_70,code=sm_70)

# 设置CUDA编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
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

# 打印构建信息
message(STATUS "CUDA版本: ${CUDA_VERSION}")
message(STATUS "CUDA编译器: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA标志: ${CUDA_NVCC_FLAGS}")
message(STATUS "构建类型: ${CMAKE_BUILD_TYPE}")