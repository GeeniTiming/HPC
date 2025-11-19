#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <tuple>

class Matrix {
private:
    std::vector<std::vector<double>> data;
    int rows;
    int cols;

public:
    Matrix(int r, int c) : rows(r), cols(c), data(r, std::vector<double>(c, 0.0)) {}
    
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    
    double& operator()(int i, int j) { 
        return data[i][j]; 
    }
    
    const double& operator()(int i, int j) const { 
        return data[i][j]; 
    }
    
    void randomInit(double min = 0.0, double max = 10.0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(min, max);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = dis(gen);
            }
        }
    }
    
    void zeroInit() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = 0.0;
            }
        }
    }
    
    static bool equals(const Matrix& a, const Matrix& b, double tolerance = 1e-6) {
        if (a.rows != b.rows || a.cols != b.cols) {
            std::cout << "矩阵维度不匹配: " << a.rows << "x" << a.cols 
                      << " vs " << b.rows << "x" << b.cols << std::endl;
            return false;
        }
        
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.cols; j++) {
                if (std::fabs(a(i, j) - b(i, j)) > tolerance) {
                    std::cout << "值不匹配 at (" << i << "," << j << "): " 
                              << a(i, j) << " vs " << b(i, j) << std::endl;
                    return false;
                }
            }
        }
        return true;
    }
};

// 串行矩阵乘法
void matrixMultiplySerial(const Matrix& A, const Matrix& B, Matrix& C) {
    if (A.getCols() != B.getRows()) {
        throw std::invalid_argument("矩阵维度不匹配");
    }
    
    int n = A.getRows();
    int m = A.getCols();
    int p = B.getCols();
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            for (int k = 0; k < m; k++) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
}

// OpenMP并行矩阵乘法 - 基础版本
double matrixMultiplyOpenMPBasic(const Matrix& A, const Matrix& B, Matrix& C, int num_threads) {
    if (A.getCols() != B.getRows()) {
        throw std::invalid_argument("矩阵维度不匹配");
    }
    
    int n = A.getRows();
    int m = A.getCols();
    int p = B.getCols();
    
    C.zeroInit();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 设置线程数
    omp_set_num_threads(num_threads);
    
    // 最简单的并行化 - 在外层循环使用OpenMP
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            for (int k = 0; k < m; k++) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    
    return duration.count();
}

// OpenMP并行矩阵乘法 - 优化版本（缓存友好）
double matrixMultiplyOpenMPOptimized(const Matrix& A, const Matrix& B, Matrix& C, int num_threads) {
    if (A.getCols() != B.getRows()) {
        throw std::invalid_argument("矩阵维度不匹配");
    }
    
    int n = A.getRows();
    int m = A.getCols();
    int p = B.getCols();
    
    C.zeroInit();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    omp_set_num_threads(num_threads);
    
    // 优化版本：调整循环顺序提高缓存命中率
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < m; k++) {
            double a_ik = A(i, k);
            for (int j = 0; j < p; j++) {
                C(i, j) += a_ik * B(k, j);
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    
    return duration.count();
}

// OpenMP并行矩阵乘法 - 分块版本（更好的缓存局部性）
double matrixMultiplyOpenMPBlocked(const Matrix& A, const Matrix& B, Matrix& C, int num_threads, int block_size = 64) {
    if (A.getCols() != B.getRows()) {
        throw std::invalid_argument("矩阵维度不匹配");
    }
    
    int n = A.getRows();
    int m = A.getCols();
    int p = B.getCols();
    
    C.zeroInit();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    omp_set_num_threads(num_threads);
    
    // 分块矩阵乘法
    #pragma omp parallel for
    for (int ii = 0; ii < n; ii += block_size) {
        for (int kk = 0; kk < m; kk += block_size) {
            for (int jj = 0; jj < p; jj += block_size) {
                // 处理当前块
                int i_end = std::min(ii + block_size, n);
                int k_end = std::min(kk + block_size, m);
                int j_end = std::min(jj + block_size, p);
                
                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        double a_ik = A(i, k);
                        for (int j = jj; j < j_end; j++) {
                            C(i, j) += a_ik * B(k, j);
                        }
                    }
                }
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    
    return duration.count();
}

// 性能测试类
class OpenMPMatrixBenchmark {
private:
    std::vector<std::tuple<int, int, int>> matrix_shapes;
    std::vector<int> thread_counts;
    
public:
    OpenMPMatrixBenchmark(const std::vector<std::tuple<int, int, int>>& shapes, 
                         const std::vector<int>& threads) 
        : matrix_shapes(shapes), thread_counts(threads) {}
    
    void runBenchmark() {
        std::cout << "OpenMP矩阵乘法性能分析" << std::endl;
        std::cout << "==============================================" << std::endl;
        std::cout << "CPU核心数: " << omp_get_num_procs() << std::endl;
        std::cout << "最大线程数: " << omp_get_max_threads() << std::endl;
        
        // 测试不同策略
        benchmarkForStrategy("基础版本", matrixMultiplyOpenMPBasic);
        benchmarkForStrategy("优化版本", matrixMultiplyOpenMPOptimized);
        benchmarkForStrategy("分块版本", [](const Matrix& A, const Matrix& B, Matrix& C, int t) {
            return matrixMultiplyOpenMPBlocked(A, B, C, t, 64);
        });
    }
    
private:
    void benchmarkForStrategy(const std::string& strategy_name, 
                             double (*multiply_func)(const Matrix&, const Matrix&, Matrix&, int)) {
        std::cout << "\n\n=== " << strategy_name << " ===" << std::endl;
        
        for (auto& shape : matrix_shapes) {
            int rows_A = std::get<0>(shape);
            int cols_A = std::get<1>(shape);
            int cols_B = std::get<2>(shape);
            
            benchmarkForShape(rows_A, cols_A, cols_B, strategy_name, multiply_func);
        }
    }
    
    void benchmarkForShape(int rows_A, int cols_A, int cols_B, 
                          const std::string& strategy_name,
                          double (*multiply_func)(const Matrix&, const Matrix&, Matrix&, int)) {
        std::cout << "\n--- 矩阵形状: A(" << rows_A << "x" << cols_A << ") × B(" 
                  << cols_A << "x" << cols_B << ") = C(" << rows_A << "x" << cols_B << ") ---" << std::endl;
        
        // 创建测试矩阵
        Matrix A(rows_A, cols_A);
        Matrix B(cols_A, cols_B);
        Matrix C_serial(rows_A, cols_B);
        Matrix C_parallel(rows_A, cols_B);
        
        A.randomInit(1.0, 5.0);
        B.randomInit(1.0, 5.0);
        
        // 串行计算基准
        double serial_time = 0.0;
        try {
            auto start = std::chrono::high_resolution_clock::now();
            matrixMultiplySerial(A, B, C_serial);
            auto end = std::chrono::high_resolution_clock::now();
            serial_time = std::chrono::duration<double>(end - start).count();
        } catch (const std::exception& e) {
            std::cout << "串行计算错误: " << e.what() << std::endl;
            return;
        }
        
        // 输出表头
        printf("%-8s %-12s %-8s %-8s %-10s\n", "线程数", "时间(秒)", "加速比", "效率", "正确性");
        printf("%-8s %-12s %-8s %-8s %-10s\n", "------", "--------", "------", "----", "------");
        
        // 单线程基准
        printf("%-8d %-12.6f %-8.3f %-8.3f %-10s\n", 1, serial_time, 1.0, 1.0, "基准");
        
        // 测试不同线程数
        for (int num_threads : thread_counts) {
            if (num_threads == 1) continue;
            
            C_parallel.zeroInit();
            double parallel_time = 0.0;
            bool correct = false;
            
            try {
                parallel_time = multiply_func(A, B, C_parallel, num_threads);
                correct = Matrix::equals(C_serial, C_parallel);
                
                double speedup = serial_time / parallel_time;
                double efficiency = speedup / num_threads * 100; // 转换为百分比
                
                printf("%-8d %-12.6f %-8.3f %-7.1f%% %-10s\n", 
                       num_threads, parallel_time, speedup, efficiency, 
                       correct ? "正确" : "错误");
                
            } catch (const std::exception& e) {
                printf("%-8d %-12s %-8s %-8s %-10s\n", 
                       num_threads, "N/A", "N/A", "N/A", e.what());
            }
        }
    }
};

// 测试不同调度策略
void testSchedulingStrategies() {
    std::cout << "\n\n=== 调度策略比较 ===" << std::endl;
    
    Matrix A(1000, 1000);
    Matrix B(1000, 1000);
    Matrix C(1000, 1000);
    
    A.randomInit(1.0, 5.0);
    B.randomInit(1.0, 5.0);
    
    int num_threads = 8;
    omp_set_num_threads(num_threads);
    
    std::vector<std::pair<std::string, std::string>> strategies = {
        {"static", "静态调度"},
        {"dynamic", "动态调度"},
        {"guided", "引导调度"},
        {"auto", "自动调度"}
    };
    
    printf("%-12s %-12s %-15s\n", "调度策略", "chunk大小", "时间(秒)");
    printf("%-12s %-12s %-15s\n", "----------", "----------", "----------");
    
    for (auto& strategy : strategies) {
        C.zeroInit();
        auto start = std::chrono::high_resolution_clock::now();
        
        #pragma omp parallel for schedule(runtime)
        for (int i = 0; i < A.getRows(); i++) {
            for (int j = 0; j < B.getCols(); j++) {
                double sum = 0.0;
                for (int k = 0; k < A.getCols(); k++) {
                    sum += A(i, k) * B(k, j);
                }
                C(i, j) = sum;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double>(end - start).count();
        
        printf("%-12s %-12s %-15.6f\n", strategy.second.c_str(), "默认", time);
    }
}

int main() {
    std::cout << "OpenMP矩阵乘法性能测试" << std::endl;
    std::cout << "编译命令: g++ -fopenmp -O2 -o openmp_matrix openmp_matrix.cpp" << std::endl;
    
    // 获取系统信息
    std::cout << "系统信息:" << std::endl;
    std::cout << "- CPU核心数: " << omp_get_num_procs() << std::endl;
    std::cout << "- 最大线程数: " << omp_get_max_threads() << std::endl;
    
    // 配置测试参数
    std::vector<std::tuple<int, int, int>> matrix_shapes = {
        // 小矩阵
        {100, 100, 100},
        // 中等矩阵
        {500, 500, 500},
        // 大矩阵
        {1000, 1000, 1000},
        // 非方阵
        {300, 100, 50},
        {800, 200, 300}
    };
    
    std::vector<int> thread_counts = {1, 2, 4, 8, 16};
    
    // 运行性能测试
    OpenMPMatrixBenchmark benchmark(matrix_shapes, thread_counts);
    benchmark.runBenchmark();
    
    // 测试调度策略
    testSchedulingStrategies();
    
    // 性能分析总结
    std::cout << "\n\n=== 性能分析总结 ===" << std::endl;
    std::cout << "1. 加速比趋势: 线程数增加时加速比提高，但效率会下降" << std::endl;
    std::cout << "2. 最佳线程数: 通常为CPU物理核心数，超线程可能带来额外收益" << std::endl;
    std::cout << "3. 矩阵大小影响: 大矩阵并行效果更好，小矩阵线程开销显著" << std::endl;
    std::cout << "4. 优化策略: 循环顺序、分块技术可以显著提高性能" << std::endl;
    std::cout << "5. 缓存效应: 优化缓存访问模式比单纯增加线程数更有效" << std::endl;
    return 0;
}