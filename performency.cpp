#include <iostream>
#include <chrono>
#include <immintrin.h>
#include <vector>
#include <thread>
#include <cmath>
#include <iomanip>
#include <cstring>

// 获取CPU信息
void print_cpu_info() {
    std::cout << "=== CPU信息 ===" << std::endl;
    
    // 通过/proc/cpuinfo获取信息 (Linux特定)
    FILE* cpuinfo = fopen("/proc/cpuinfo", "rb");
    if (cpuinfo) {
        char* arg = 0;
        size_t size = 0;
        while (getdelim(&arg, &size, 0, cpuinfo) != -1) {
            if (strstr(arg, "model name")) {
                std::cout << "CPU型号: " << (arg + strlen("model name") + 2);
                break;
            }
        }
        free(arg);
        fclose(cpuinfo);
    }
    
    std::cout << "CPU核心数: " << std::thread::hardware_concurrency() << std::endl;
    std::cout << "=================" << std::endl;
}

// 使用AVX指令集进行双精度浮点运算测试
double test_avx_dp(int iterations) {
    // 初始化AVX寄存器
    __m256d a = _mm256_set1_pd(1.0);
    __m256d b = _mm256_set1_pd(1.0);
    __m256d c = _mm256_set1_pd(1.0);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        // 使用FMA指令 (融合乘加)
        // 每次执行4次双精度乘加操作 = 8次浮点运算
        c = _mm256_fmadd_pd(a, b, c);
        
        // 防止编译器优化掉循环
        asm volatile("" : "+x"(c) : : "memory");
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    // 确保结果被使用
    volatile double result = ((double*)&c)[0];
    (void)result; // 防止未使用变量警告
    
    return diff.count();
}

// 使用AVX指令集进行单精度浮点运算测试
double test_avx_sp(int iterations) {
    // 初始化AVX寄存器
    __m256 a = _mm256_set1_ps(1.0f);
    __m256 b = _mm256_set1_ps(1.0f);
    __m256 c = _mm256_set1_ps(1.0f);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        // 使用FMA指令 (融合乘加)
        // 每次执行8次单精度乘加操作 = 16次浮点运算
        c = _mm256_fmadd_ps(a, b, c);
        
        // 防止编译器优化掉循环
        asm volatile("" : "+x"(c) : : "memory");
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    // 确保结果被使用
    volatile float result = ((float*)&c)[0];
    (void)result; // 防止未使用变量警告
    
    return diff.count();
}

// 多线程测试函数
void thread_test(int thread_id, int iterations, double& result) {
    __m256d a = _mm256_set1_pd(1.0);
    __m256d b = _mm256_set1_pd(1.0);
    __m256d c = _mm256_set1_pd(1.0);
    
    for (int i = 0; i < iterations; ++i) {
        c = _mm256_fmadd_pd(a, b, c);
        asm volatile("" : "+x"(c) : : "memory");
    }
    
    volatile double res = ((double*)&c)[0];
    result = res;
}

// 多线程性能测试
double test_multithreaded(int iterations_per_thread, int num_threads) {
    std::vector<std::thread> threads;
    std::vector<double> results(num_threads, 0.0);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(thread_test, i, iterations_per_thread, std::ref(results[i]));
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    return diff.count();
}

int main() {
    std::cout << "=== CPU GFLOPS性能测试 ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    
    print_cpu_info();
    
    // 测试参数
    const int iterations = 100000000; // 1亿次迭代
    const int threads = std::thread::hardware_concurrency();
    
    std::cout << "测试参数:" << std::endl;
    std::cout << "迭代次数: " << iterations << std::endl;
    std::cout << "线程数: " << threads << std::endl;
    std::cout << "=================" << std::endl;
    
    // 双精度测试
    double time_dp = test_avx_dp(iterations);
    double flops_dp = (iterations * 8.0) / time_dp; // 每次迭代8次浮点运算
    double gflops_dp = flops_dp / 1e9;
    
    std::cout << "双精度测试结果:" << std::endl;
    std::cout << "时间: " << time_dp << " 秒" << std::endl;
    std::cout << "性能: " << gflops_dp << " GFLOPS" << std::endl;
    std::cout << "-----------------" << std::endl;
    
    // 单精度测试
    double time_sp = test_avx_sp(iterations);
    double flops_sp = (iterations * 16.0) / time_sp; // 每次迭代16次浮点运算
    double gflops_sp = flops_sp / 1e9;
    
    std::cout << "单精度测试结果:" << std::endl;
    std::cout << "时间: " << time_sp << " 秒" << std::endl;
    std::cout << "性能: " << gflops_sp << " GFLOPS" << std::endl;
    std::cout << "-----------------" << std::endl;
    
    // 多线程测试
    if (threads > 1) {
        int iterations_per_thread = iterations / 4; // 减少每个线程的迭代次数
        double time_mt = test_multithreaded(iterations_per_thread, threads);
        double flops_mt = (iterations_per_thread * threads * 8.0) / time_mt;
        double gflops_mt = flops_mt / 1e9;
        
        std::cout << "多线程测试结果 (" << threads << " 线程):" << std::endl;
        std::cout << "时间: " << time_mt << " 秒" << std::endl;
        std::cout << "性能: " << gflops_mt << " GFLOPS" << std::endl;
        std::cout << "-----------------" << std::endl;
    }
    
    std::cout << "测试完成!" << std::endl;
    
    return 0;
}