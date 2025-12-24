#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static double frand() { return rand() / (double)RAND_MAX; }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2) {
        if (rank == 0) printf("usage: %s N [seed]\n", argv[0]);
        MPI_Finalize();
        return 0;
    }

    int N = atoi(argv[1]);
    unsigned int seed = (argc >= 3) ? (unsigned int)atoi(argv[2]) : 42u;
    if (N <= 0) {
        if (rank == 0) printf("N must be > 0\n");
        MPI_Finalize();
        return 0;
    }

    int* rows = (int*)malloc((size_t)size * sizeof(int));
    int* disp_rows = (int*)malloc((size_t)size * sizeof(int));
    for (int p = 0; p < size; ++p) rows[p] = N / size + (p < (N % size));
    disp_rows[0] = 0;
    for (int p = 1; p < size; ++p) disp_rows[p] = disp_rows[p - 1] + rows[p - 1];
    int local_rows = rows[rank];

    double* A = NULL;
    double* x = (double*)malloc((size_t)N * sizeof(double));
    double* y = NULL;
    size_t a_loc_elems = (size_t)local_rows * (size_t)N;
    double* A_local = (double*)malloc((a_loc_elems ? a_loc_elems : 1) * sizeof(double));
    double* y_local = (double*)malloc((local_rows ? local_rows : 1) * sizeof(double));

    if (rank == 0) {
        srand(seed);
        A = (double*)malloc((size_t)N * (size_t)N * sizeof(double));
        y = (double*)malloc((size_t)N * sizeof(double));
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) A[(size_t)i * (size_t)N + j] = frand();
            x[i] = frand();
        }
    }

    double seq_time = 0.0;
    double* y_seq = NULL;
    if (rank == 0) {
        y_seq = (double*)malloc((size_t)N * sizeof(double));
        double t0 = MPI_Wtime();
        for (int i = 0; i < N; ++i) {
            double s = 0.0;
            for (int j = 0; j < N; ++j) s += A[(size_t)i * (size_t)N + j] * x[j];
            y_seq[i] = s;
        }
        seq_time = MPI_Wtime() - t0;
    }
    MPI_Bcast(&seq_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int* sendcounts = (int*)malloc((size_t)size * sizeof(int));
    int* displs = (int*)malloc((size_t)size * sizeof(int));
    for (int p = 0; p < size; ++p) sendcounts[p] = rows[p] * N;
    displs[0] = 0;
    for (int p = 1; p < size; ++p) displs[p] = displs[p - 1] + sendcounts[p - 1];

    double t0p = MPI_Wtime();
    if (rank != 0) memset(x, 0, (size_t)N * sizeof(double));
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, A_local, local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_rows; ++i) {
        double s = 0.0;
        size_t base = (size_t)i * (size_t)N;
        for (int j = 0; j < N; ++j) s += A_local[base + j] * x[j];
        y_local[i] = s;
    }

    if (rank == 0 && y == NULL) y = (double*)malloc((size_t)N * sizeof(double));
    MPI_Gatherv(y_local, local_rows, MPI_DOUBLE, y, rows, disp_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double par_time = MPI_Wtime() - t0p;

    if (rank == 0) {
        double max_err = 0.0;
        for (int i = 0; i < N; ++i) {
            double d = fabs(y[i] - y_seq[i]);
            if (d > max_err) max_err = d;
        }
        double speedup = seq_time / par_time;
        double efficiency = speedup / (double)size;
        printf("N=%d procs=%d time=%.6f speedup=%.6f efficiency=%.6f max_error=%.3e\n", N, size, par_time, speedup, efficiency, max_err);
    }

    free(rows);
    free(disp_rows);
    free(sendcounts);
    free(displs);
    free(A_local);
    free(y_local);
    free(x);
    if (rank == 0) { free(A); free(y); free(y_seq); }
    MPI_Finalize();
    return 0;
}
