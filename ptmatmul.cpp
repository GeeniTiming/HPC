#include <iostream>
#include <vector>
#include <pthread.h>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <tuple>

class Matrix {
private:
    std::vector<std::vector<double>> data;
    int rows;
    int cols;

public:
    Matrix(int r, int c) : rows(r), cols(c), data(r, std::vector<double>(c, 0.0)) {}
    
    // 拷贝构造函数
    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), data(other.data) {}
    
    // 赋值运算符
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            rows = other.rows;
            cols = other.cols;
            data = other.data;
        }
        return *this;
    }
    
    int getRows() const { return rows; }
    int getCols() const { return cols; }
    
    double& operator()(int i, int j) { 
        // 添加边界检查
        if (i < 0 || i >= rows || j < 0 || j >= cols) {
            throw std::out_of_range("Matrix index out of range");
        }
        return data[i][j]; 
    }
    
    const double& operator()(int i, int j) const { 
        if (i < 0 || i >= rows || j < 0 || j >= cols) {
            throw std::out_of_range("Matrix index out of range");
        }
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
    
    void print() const {
        int display_rows = std::min(rows, 5);
        int display_cols = std::min(cols, 5);
        
        for (int i = 0; i < display_rows; i++) {
            for (int j = 0; j < display_cols; j++) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) << data[i][j];
            }
            if (cols > display_cols) std::cout << " ...";
            std::cout << std::endl;
        }
        if (rows > display_rows) std::cout << " ... (" << rows << "x" << cols << ")" << std::endl;
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

// 线程参数结构
struct ThreadArgs {
    int thread_id;
    int num_threads;
    const Matrix* A;
    const Matrix* B;
    Matrix* C;
};

// 串行矩阵乘法
void matrixMultiplySerial(const Matrix& A, const Matrix& B, Matrix& C) {
    if (A.getCols() != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
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

// 线程函数 - 按行分块（支持非方阵）
void* matrixMultiplyThreadRow(void* arg) {
    ThreadArgs* args = static_cast<ThreadArgs*>(arg);
    const Matrix& A = *(args->A);
    const Matrix& B = *(args->B);
    Matrix& C = *(args->C);
    int thread_id = args->thread_id;
    int num_threads = args->num_threads;
    
    int n = A.getRows();
    int m = A.getCols();
    int p = B.getCols();
    
    // 简单的按行分块，确保负载均衡
    int rows_per_thread = n / num_threads;
    int extra_rows = n % num_threads;
    
    int start_row = thread_id * rows_per_thread + std::min(thread_id, extra_rows);
    int end_row = start_row + rows_per_thread + (thread_id < extra_rows ? 1 : 0);
    
    // 确保不越界
    end_row = std::min(end_row, n);
    
    // 计算分配的行
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            for (int k = 0; k < m; k++) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
    
    return nullptr;
}

// 线程函数 - 按元素分块（支持非方阵）
void* matrixMultiplyThreadElement(void* arg) {
    ThreadArgs* args = static_cast<ThreadArgs*>(arg);
    const Matrix& A = *(args->A);
    const Matrix& B = *(args->B);
    Matrix& C = *(args->C);
    int thread_id = args->thread_id;
    int num_threads = args->num_threads;
    int n = A.getRows();
    int p = B.getCols();
    int m = A.getCols();
    int total_elements = n * p;
    int elements_per_thread = total_elements / num_threads;
    int extra_elements = total_elements % num_threads;
    int start_element = thread_id * elements_per_thread + std::min(thread_id, extra_elements);
    int end_element = start_element + elements_per_thread + (thread_id < extra_elements ? 1 : 0);
    end_element = std::min(end_element, total_elements);
    for (int elem = start_element; elem < end_element; elem++) {
        int i = elem / p;  // 行索引
        int j = elem % p;  // 列索引
        double sum = 0.0;
        for (int k = 0; k < m; k++) {
            sum += A(i, k) * B(k, j);
        }
        C(i, j) = sum;
    }
    return nullptr;
}

// 并行矩阵乘法
double matrixMultiplyParallel(const Matrix& A, const Matrix& B, Matrix& C, int num_threads, bool use_element_partition = true) {
    if (A.getCols() != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }
    if (C.getRows() != A.getRows() || C.getCols() != B.getCols()) {
        throw std::invalid_argument("Result matrix has wrong dimensions");
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    if (num_threads == 1) {
        matrixMultiplySerial(A, B, C);
    } else {
        std::vector<pthread_t> threads(num_threads);
        std::vector<ThreadArgs> thread_args(num_threads);
        for (int i = 0; i < num_threads; i++) {
            thread_args[i].thread_id = i;
            thread_args[i].num_threads = num_threads;
            thread_args[i].A = &A;
            thread_args[i].B = &B;
            thread_args[i].C = &C;
        }
        void* (*thread_func)(void*) = use_element_partition ? 
            matrixMultiplyThreadElement : matrixMultiplyThreadRow;
        
        // 创建线程
        for (int i = 0; i < num_threads; i++) {
            if (pthread_create(&threads[i], nullptr, thread_func, &thread_args[i]) != 0) {
                for (int j = 0; j < i; j++) {
                    pthread_join(threads[j], nullptr);
                }
                throw std::runtime_error("Failed to create thread");
            }
        }
        for (int i = 0; i < num_threads; i++) {
            if (pthread_join(threads[i], nullptr) != 0) {
                throw std::runtime_error("Failed to join thread");
            }
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    return duration.count();
}

// 性能测试类 - 支持非方阵
class MatrixMultiplicationBenchmark {
private:
    std::vector<std::tuple<int, int, int>> matrix_shapes; // (rows_A, cols_A, cols_B)
    std::vector<int> thread_counts;
    
public:
    MatrixMultiplicationBenchmark(const std::vector<std::tuple<int, int, int>>& shapes, 
                                 const std::vector<int>& threads) 
        : matrix_shapes(shapes), thread_counts(threads) {}
    
    void runBenchmark() {
        std::cout << "多线程矩阵乘法性能分析 (支持非方阵)" << std::endl;
        std::cout << "==============================================" << std::endl;
        
        // 测试两种分区策略
        benchmarkForStrategy("按元素分区", true);
        benchmarkForStrategy("按行分区", false);
    }
    
private:
    void benchmarkForStrategy(const std::string& strategy_name, bool use_element_partition) {
        std::cout << "\n=== " << strategy_name << " ===" << std::endl;
        
        for (auto& shape : matrix_shapes) {
            int rows_A = std::get<0>(shape);
            int cols_A = std::get<1>(shape);
            int cols_B = std::get<2>(shape);
            
            benchmarkForShape(rows_A, cols_A, cols_B, strategy_name, use_element_partition);
        }
    }
    
    void benchmarkForShape(int rows_A, int cols_A, int cols_B, const std::string& strategy_name, bool use_element_partition) {
        std::cout << "\n--- 矩阵形状: A(" << rows_A << "x" << cols_A << ") × B(" 
                  << cols_A << "x" << cols_B << ") = C(" << rows_A << "x" << cols_B << ") ---" << std::endl;
        std::cout << "线程数" 
                  << '\t' << "时间(秒)" 
                  << '\t' << "加速比" 
                  << '\t' << "效率" 
                  << '\t' << "正确性" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        try {
            // 创建测试矩阵
            Matrix A(rows_A, cols_A);
            Matrix B(cols_A, cols_B);
            Matrix C_serial(rows_A, cols_B);
            
            A.randomInit(1.0, 5.0);  // 使用较小的范围避免数值问题
            B.randomInit(1.0, 5.0);
            C_serial.zeroInit();
            
            // 串行计算基准
            double serial_time = 0.0;
            auto start = std::chrono::high_resolution_clock::now();
            matrixMultiplySerial(A, B, C_serial);
            auto end = std::chrono::high_resolution_clock::now();
            serial_time = std::chrono::duration<double>(end - start).count();
            
            std::cout  << 1 
                      << '\t' << std::fixed << std::setprecision(6) << serial_time
                      << '\t' << std::fixed << std::setprecision(3) << 1.0
                      << '\t' << std::fixed << std::setprecision(3) << 1.0
                      << '\t' << "基准" << std::endl;
            
            // 测试不同线程数
            for (int num_threads : thread_counts) {
                if (num_threads == 1) continue;
                
                Matrix C_parallel(rows_A, cols_B);
                C_parallel.zeroInit();
                double parallel_time = 0.0;
                bool correct = false;
                
                try {
                    parallel_time = matrixMultiplyParallel(A, B, C_parallel, num_threads, use_element_partition);
                    correct = Matrix::equals(C_serial, C_parallel);
                    
                    double speedup = serial_time / parallel_time;
                    double efficiency = speedup / num_threads;
                    
                    std::cout << num_threads
                              << '\t' << std::fixed << std::setprecision(6) << parallel_time
                              << '\t' << std::fixed << std::setprecision(3) << speedup
                              << '\t' << std::fixed << std::setprecision(3) << efficiency
                              << '\t' << (correct ? "正确" : "错误") << std::endl;
                              
                } catch (const std::exception& e) {
                    std::cout << num_threads
                              << '\t' << "N/A"
                              << '\t' << "N/A" 
                              << '\t' << "N/A"
                              << '\t' << "错误: " << e.what() << std::endl;
                }
            }
            
        } catch (const std::exception& e) {
            std::cout << "测试错误: " << e.what() << std::endl;
        }
    }
};

int main() {
    std::cout << "开始矩阵乘法测试..." << std::endl;
    
    // 先测试小矩阵确保基本功能正常
    std::cout << "\n=== 基本功能测试 (非方阵) ===" << std::endl;
    try {
        // 测试非方阵
        Matrix A(2, 3);
        Matrix B(3, 4);
        Matrix C1(2, 4);
        Matrix C2(2, 4);
        
        // 初始化测试矩阵
        double counter = 1.0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                A(i, j) = counter++;
            }
        }
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                B(i, j) = counter++;
            }
        }
        
        matrixMultiplySerial(A, B, C1);
        matrixMultiplyParallel(A, B, C2, 2, true);
        
        std::cout << "非方阵测试: " << (Matrix::equals(C1, C2) ? "通过" : "失败") << std::endl;
        
        std::cout << "矩阵 A (2x3):" << std::endl;
        A.print();
        std::cout << "矩阵 B (3x4):" << std::endl;
        B.print();
        std::cout << "结果 C (2x4):" << std::endl;
        C1.print();
        
    } catch (const std::exception& e) {
        std::cout << "基本功能测试失败: " << e.what() << std::endl;
        return 1;
    }
    
    // 配置测试参数（包含各种非方阵情况）
    std::vector<std::tuple<int, int, int>> matrix_shapes = {
        // 方阵
        {100, 100, 100},
        // 行数大于列数
        {200, 50, 100},
        {300, 100, 50},
        // 列数大于行数  
        {50, 200, 100},
        {100, 300, 200},
        // 一般非方阵
        {150, 200, 250},
        {80, 120, 160}
    };
    
    std::vector<int> thread_counts = {1, 2, 4, 8};
    
    // 运行性能测试
    try {
        MatrixMultiplicationBenchmark benchmark(matrix_shapes, thread_counts);
        benchmark.runBenchmark();
    } catch (const std::exception& e) {
        std::cout << "性能测试失败: " << e.what() << std::endl;
        return 1;
    }
    
    // 额外演示一些非方阵例子
    std::cout << "\n=== 非方阵演示 ===" << std::endl;
    
    // 演示1: 瘦高矩阵 × 矮宽矩阵
    try {
        Matrix A(5, 2);  // 瘦高
        Matrix B(2, 6);  // 矮宽
        Matrix C(5, 6);
        
        A.randomInit(1, 3);
        B.randomInit(1, 3);
        
        matrixMultiplyParallel(A, B, C, 2);
        
        std::cout << "瘦高矩阵 × 矮宽矩阵:" << std::endl;
        std::cout << "A (" << A.getRows() << "x" << A.getCols() << "):" << std::endl;
        A.print();
        std::cout << "B (" << B.getRows() << "x" << B.getCols() << "):" << std::endl;
        B.print();
        std::cout << "结果 C (" << C.getRows() << "x" << C.getCols() << "):" << std::endl;
        C.print();
    } catch (const std::exception& e) {
        std::cout << "演示1错误: " << e.what() << std::endl;
    }
    
    // 演示2: 行向量 × 矩阵
    try {
        Matrix A(1, 4);  // 行向量
        Matrix B(4, 3);  // 矩阵
        Matrix C(1, 3);  // 结果也是行向量
        
        A.randomInit(1, 3);
        B.randomInit(1, 3);
        
        matrixMultiplyParallel(A, B, C, 2);
        
        std::cout << "\n行向量 × 矩阵:" << std::endl;
        std::cout << "A (" << A.getRows() << "x" << A.getCols() << "):" << std::endl;
        A.print();
        std::cout << "B (" << B.getRows() << "x" << B.getCols() << "):" << std::endl;
        B.print();
        std::cout << "结果 C (" << C.getRows() << "x" << C.getCols() << "):" << std::endl;
        C.print();
    } catch (const std::exception& e) {
        std::cout << "演示2错误: " << e.what() << std::endl;
    }
    
    std::cout << "\n测试完成!" << std::endl;
    return 0;
}