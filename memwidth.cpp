#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

// 测试内存读取带宽
double test_memory_read_bandwidth(size_t data_size, int iterations) {
    // 分配大内存块
    std::vector<char> buffer(data_size);
    
    // 初始化数据（确保内存实际分配）
    for (size_t i = 0; i < data_size; ++i) {
        buffer[i] = static_cast<char>(i % 256);
    }
    
    volatile char sink; // 用于防止优化
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // 多次读取整个内存块
    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < data_size; ++i) {
            sink = buffer[i]; // 读取操作
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    (void)sink; // 防止未使用变量警告
    
    // 计算总数据量 (字节)
    double total_bytes = data_size * iterations;
    
    // 计算带宽 (GB/s)
    double bandwidth_gbs = (total_bytes / diff.count()) / (1024 * 1024 * 1024);
    
    return bandwidth_gbs;
}

// 测试内存写入带宽
double test_memory_write_bandwidth(size_t data_size, int iterations) {
    // 分配大内存块
    std::vector<char> buffer(data_size);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // 多次写入整个内存块
    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < data_size; ++i) {
            buffer[i] = static_cast<char>(i % 256); // 写入操作
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    // 确保buffer被使用（防止优化）
    volatile char check = buffer[0];
    (void)check;
    
    // 计算总数据量 (字节)
    double total_bytes = data_size * iterations;
    
    // 计算带宽 (GB/s)
    double bandwidth_gbs = (total_bytes / diff.count()) / (1024 * 1024 * 1024);
    
    return bandwidth_gbs;
}

// 测试内存读写带宽（同时读写）
double test_memory_readwrite_bandwidth(size_t data_size, int iterations) {
    // 分配两个内存块
    std::vector<char> src(data_size);
    std::vector<char> dst(data_size);
    
    // 初始化源数据
    for (size_t i = 0; i < data_size; ++i) {
        src[i] = static_cast<char>(i % 256);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // 多次复制整个内存块（同时读写）
    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < data_size; ++i) {
            dst[i] = src[i]; // 读取+写入操作
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    // 确保dst被使用（防止优化）
    volatile char check = dst[0];
    (void)check;
    
    // 计算总数据量 (字节) - 读取和写入各data_size
    double total_bytes = 2 * data_size * iterations;
    
    // 计算带宽 (GB/s)
    double bandwidth_gbs = (total_bytes / diff.count()) / (1024 * 1024 * 1024);
    
    return bandwidth_gbs;
}

int main() {
    std::cout << "=== 内存带宽测试 ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    
    // 测试参数
    const size_t data_size = 100 * 1024 * 1024; // 100 MB
    const int iterations = 10;
    
    std::cout << "测试参数:" << std::endl;
    std::cout << "数据大小: " << data_size / (1024 * 1024) << " MB" << std::endl;
    std::cout << "迭代次数: " << iterations << std::endl;
    std::cout << "=================" << std::endl;
    
    // 测试读取带宽
    double read_bw = test_memory_read_bandwidth(data_size, iterations);
    std::cout << "读取带宽: " << read_bw << " GB/s" << std::endl;
    
    // 测试写入带宽
    double write_bw = test_memory_write_bandwidth(data_size, iterations);
    std::cout << "写入带宽: " << write_bw << " GB/s" << std::endl;
    
    // 测试读写带宽
    double readwrite_bw = test_memory_readwrite_bandwidth(data_size, iterations);
    std::cout << "读写带宽: " << readwrite_bw << " GB/s" << std::endl;
    
    std::cout << "=================" << std::endl;
    std::cout << "测试完成!" << std::endl;
    
    return 0;
}