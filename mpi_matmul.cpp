#include <iostream>
#include <mpi.h>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>

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

void matrixMultiplyMPI(const Matrix& A, const Matrix& B, Matrix& C, int rank, int size)
{
    int n = A.getRows();
    int m = A.getCols();
    int p = B.getCols();
    int rows_per_proc = n / size;
    int remaining = n % size;
    int local_rows = (rank < remaining) ? rows_per_proc + 1 : rows_per_proc;
    int start_row = rank * rows_per_proc + std::min(rank, remaining);
    Matrix local_A(local_rows, m);
    Matrix local_B(m, p);
    Matrix local_C(local_rows, p);
    std::vector<double> b_flat(m * p);
    if (rank == 0)
    {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                b_flat[i * p + j] = B(i, j);
            }
        }
    }
    MPI_Bcast(b_flat.data(), m * p, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            local_B(i, j) = b_flat[i * p + j];
        }
    }
    if (rank == 0)
    {
        for (size_t i = 0; i < local_rows; i++)
        {
            for (size_t j = 0; j < m; j++)
            {
                local_A(i, j) = A(i, j);
            }
        }
        int current_offset = local_rows;
        for (size_t dest = 1; dest < size; dest++)
        {
            int dest_rows = (dest < remaining) ? rows_per_proc + 1 : rows_per_proc;
            std::vector<double> send_buffer(dest_rows * m);
            for (size_t i = 0; i < dest_rows; i++)
            {
                for (size_t j = 0; j < m; j++)
                {
                    send_buffer[i * m + j] = A(current_offset + i, j);
                }
            }
            MPI_Send(send_buffer.data(), dest_rows * m, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            current_offset += dest_rows;
        }
    }
    else
    {
        std::vector<double> recv_buffer(local_rows * m);
        MPI_Recv(recv_buffer.data(), local_rows * m, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (size_t i = 0; i < local_rows; i++)
        {
            for (size_t j = 0; j < m; j++)
            {
                local_A(i, j) = recv_buffer[i * m + j];
            }
        }
    }
    for (size_t i = 0; i < local_rows; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            double sum = 0.0;
            for (size_t k = 0; k < m; k++)
            {
                sum += local_A(i, k) * local_B(k, j);
            }
            local_C(i, j) = sum;
        }
    }
    if (rank == 0)
    {
        for (size_t i = 0; i < local_rows; i++)
        {
            for (size_t j = 0; j < p; j++)
            {
                C(i, j) = local_C(i, j);
            }
        }
        int current_offset = local_rows;
        for (size_t src = 1; src < size; src++)
        {
            int src_rows = (src < remaining) ? rows_per_proc + 1 : rows_per_proc;
            std::vector<double> recv_buffer(src_rows * p);
            MPI_Recv(recv_buffer.data(), src_rows * p, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (size_t i = 0; i < src_rows; i++)
            {
                for (size_t j = 0; j < p; j++)
                {
                    C(current_offset + i, j) = recv_buffer[i * p + j];
                }
            }
            current_offset += src_rows;
        }
    }
    else
    {
        std::vector<double> send_buffer(local_rows * p);
        for (size_t i = 0; i < local_rows; i++)
        {
            for (size_t j = 0; j < p; j++)
            {
                send_buffer[i * p + j] = local_C(i, j);
            }
        }
        MPI_Send(send_buffer.data(), local_rows * p, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<std::tuple<int, int, int>> matrix_configs = {
        {1000, 1000, 1000},
        {5000, 5000, 1},
        {10000, 10000, 1}
    };
    if (rank == 0)
    {
        
    }    
    return 0;
}
