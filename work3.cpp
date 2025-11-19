#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <string.h>

// 生成随机矩阵
void generate_matrix(double *matrix, int rows, int cols) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = (double)rand() / RAND_MAX * 100.0;
        }
    }
}

// 统一的串行矩阵乘法 (支持矩阵×矩阵和矩阵×向量)
void matrix_mult_serial(double *A, double *B, double *C, 
                       int m, int n, int p) {
    // A: m×n, B: n×p, C: m×p
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i * p + j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

// 统一的并行矩阵乘法 - 外层循环并行
void matrix_mult_parallel_outer(double *A, double *B, double *C, 
                               int m, int n, int p, int num_threads) {
    // A: m×n, B: n×p, C: m×p
    #pragma omp parallel for num_threads(num_threads) collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

// 统一的并行矩阵乘法 - 分块优化
void matrix_mult_parallel_block(double *A, double *B, double *C, 
                               int m, int n, int p, int num_threads, int block_size) {
    #pragma omp parallel for num_threads(num_threads) collapse(2)
    for (int i = 0; i < m; i += block_size) {
        for (int j = 0; j < p; j += block_size) {
            for (int k = 0; k < n; k += block_size) {
                // 处理块
                for (int ii = i; ii < i + block_size && ii < m; ii++) {
                    for (int jj = j; jj < j + block_size && jj < p; jj++) {
                        double sum = 0.0;
                        for (int kk = k; kk < k + block_size && kk < n; kk++) {
                            sum += A[ii * n + kk] * B[kk * p + jj];
                        }
                        C[ii * p + jj] += sum;
                    }
                }
            }
        }
    }
}

// 验证结果是否相等
int verify_results(double *result1, double *result2, int size, double tolerance) {
    for (int i = 0; i < size; i++) {
        if (fabs(result1[i] - result2[i]) > tolerance) {
            printf("验证失败! 位置 %d: %f != %f\n", i, result1[i], result2[i]);
            return 0;
        }
    }
    return 1;
}

// 性能测试函数
void performance_test() {
    int sizes[] = {500, 1000, 1500};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int thread_counts[] = {1, 2, 4, 8};
    int num_threads = sizeof(thread_counts) / sizeof(thread_counts[0]);
    
    printf("统一矩阵乘法性能测试 (矩阵×矩阵):\n");
    printf("================================\n\n");
    
    for (int s = 0; s < num_sizes; s++) {
        int size = sizes[s];
        printf("矩阵大小: %d × %d\n", size, size);
        printf("----------------------------------------\n");
        
        // 分配内存
        double *A = (double*)malloc(size * size * sizeof(double));
        double *B = (double*)malloc(size * size * sizeof(double));
        double *C_serial = (double*)malloc(size * size * sizeof(double));
        double *C_parallel = (double*)malloc(size * size * sizeof(double));
        
        // 生成测试数据
        generate_matrix(A, size, size);
        generate_matrix(B, size, size);
        
        // 串行计算
        double start_time = omp_get_wtime();
        matrix_mult_serial(A, B, C_serial, size, size, size);
        double serial_time = omp_get_wtime() - start_time;
        
        printf("串行计算时间: %.4f 秒\n", serial_time);
        
        // 并行计算 - 不同线程数
        for (int t = 0; t < num_threads; t++) {
            int threads = thread_counts[t];
            
            // 外层循环并行
            memset(C_parallel, 0, size * size * sizeof(double));
            start_time = omp_get_wtime();
            matrix_mult_parallel_outer(A, B, C_parallel, size, size, size, threads);
            double parallel_time_outer = omp_get_wtime() - start_time;
            
            int valid_outer = verify_results(C_serial, C_parallel, size * size, 1e-6);
            
            // 分块并行 (块大小=32)
            memset(C_parallel, 0, size * size * sizeof(double));
            start_time = omp_get_wtime();
            matrix_mult_parallel_block(A, B, C_parallel, size, size, size, threads, 32);
            double parallel_time_block = omp_get_wtime() - start_time;
            
            int valid_block = verify_results(C_serial, C_parallel, size * size, 1e-6);
            
            printf("线程数 %d:\n", threads);
            printf("  外层并行: %.4f 秒 (加速比: %.2fx, 验证: %s)\n", 
                   parallel_time_outer, serial_time / parallel_time_outer,
                   valid_outer ? "通过" : "失败");
            printf("  分块并行: %.4f 秒 (加速比: %.2fx, 验证: %s)\n", 
                   parallel_time_block, serial_time / parallel_time_block,
                   valid_block ? "通过" : "失败");
        }
        printf("\n");
        
        free(A);
        free(B);
        free(C_serial);
        free(C_parallel);
    }
}

// 矩阵向量乘法测试 (使用统一的矩阵乘法函数)
void matrix_vector_test() {
    printf("矩阵向量乘法测试 (作为矩阵×矩阵的特例):\n");
    printf("=====================================\n\n");
    
    int matrix_rows = 10000;
    int matrix_cols = 10000;
    int vector_cols = 1;  // 向量被视为 matrix_cols × 1 的矩阵
    
    int thread_counts[] = {1, 2, 4, 8};
    int num_threads = sizeof(thread_counts) / sizeof(thread_counts[0]);
    
    // 分配内存
    double *matrix = (double*)malloc(matrix_rows * matrix_cols * sizeof(double));
    double *vector = (double*)malloc(matrix_cols * vector_cols * sizeof(double));
    double *result_serial = (double*)malloc(matrix_rows * vector_cols * sizeof(double));
    double *result_parallel = (double*)malloc(matrix_rows * vector_cols * sizeof(double));
    
    // 生成测试数据
    generate_matrix(matrix, matrix_rows, matrix_cols);
    generate_matrix(vector, matrix_cols, vector_cols);  // 向量作为列向量
    
    // 串行计算
    double start_time = omp_get_wtime();
    matrix_mult_serial(matrix, vector, result_serial, matrix_rows, matrix_cols, vector_cols);
    double serial_time = omp_get_wtime() - start_time;
    
    printf("矩阵大小: %d × %d, 向量大小: %d × %d\n", 
           matrix_rows, matrix_cols, matrix_cols, vector_cols);
    printf("串行计算时间: %.4f 秒\n", serial_time);
    
    // 并行计算 - 不同线程数
    for (int t = 0; t < num_threads; t++) {
        int threads = thread_counts[t];
        
        memset(result_parallel, 0, matrix_rows * vector_cols * sizeof(double));
        start_time = omp_get_wtime();
        matrix_mult_parallel_outer(matrix, vector, result_parallel, 
                                 matrix_rows, matrix_cols, vector_cols, threads);
        double parallel_time = omp_get_wtime() - start_time;
        
        int valid = verify_results(result_serial, result_parallel, matrix_rows, 1e-6);
        
        printf("线程数 %d: 并行时间 %.4f 秒 (加速比: %.2fx, 验证: %s)\n", 
               threads, parallel_time, serial_time / parallel_time,
               valid ? "通过" : "失败");
    }
    printf("\n");
    
    free(matrix);
    free(vector);
    free(result_serial);
    free(result_parallel);
}

// 测试不同向量方向 (行向量 vs 列向量)
void vector_orientation_test() {
    printf("不同向量方向测试:\n");
    printf("================\n\n");
    
    int matrix_rows = 5000;
    int matrix_cols = 5000;
    int thread_count = 4;
    
    // 测试1: 矩阵 × 列向量 (matrix_cols × 1)
    printf("测试1: 矩阵 × 列向量\n");
    double *matrix = (double*)malloc(matrix_rows * matrix_cols * sizeof(double));
    double *col_vector = (double*)malloc(matrix_cols * 1 * sizeof(double));
    double *result_col = (double*)malloc(matrix_rows * 1 * sizeof(double));
    
    generate_matrix(matrix, matrix_rows, matrix_cols);
    generate_matrix(col_vector, matrix_cols, 1);
    
    double start_time = omp_get_wtime();
    matrix_mult_parallel_outer(matrix, col_vector, result_col, 
                             matrix_rows, matrix_cols, 1, thread_count);
    double time_col = omp_get_wtime() - start_time;
    printf("列向量计算时间: %.4f 秒\n", time_col);
    
    // 测试2: 行向量 × 矩阵 (1 × matrix_rows) × (matrix_rows × matrix_cols)
    printf("\n测试2: 行向量 × 矩阵\n");
    double *row_vector = (double*)malloc(1 * matrix_rows * sizeof(double));
    double *result_row = (double*)malloc(1 * matrix_cols * sizeof(double));
    
    generate_matrix(row_vector, 1, matrix_rows);
    
    start_time = omp_get_wtime();
    matrix_mult_parallel_outer(row_vector, matrix, result_row, 
                             1, matrix_rows, matrix_cols, thread_count);
    double time_row = omp_get_wtime() - start_time;
    printf("行向量计算时间: %.4f 秒\n", time_row);
    
    free(matrix);
    free(col_vector);
    free(result_col);
    free(row_vector);
    free(result_row);
}

int main() {
    printf("OpenMP 统一并行矩阵乘法性能测试\n");
    printf("===============================\n\n");
    
    // 设置随机种子
    srand(42);
    
    // 测试矩阵向量乘法 (作为特例)
    matrix_vector_test();
    
    // 测试不同向量方向
    vector_orientation_test();
    
    // 测试完整矩阵乘法
    performance_test();
    
    return 0;
}