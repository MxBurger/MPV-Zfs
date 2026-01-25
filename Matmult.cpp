#include <mpi.h>
#include <iostream>
#include <random>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    
    int M = 1000, N = 100;
    int rows_per_proc = M / comm_size;
    
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Schritt 1: Broadcast x
    double* x = new double[N];
    if (rank == 0) {
        for (int i = 0; i < N; i++) x[i] = dist(gen);
    }
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Schritt 2: Scatter A
    double* A = nullptr;
    if (rank == 0) {
        A = new double[M * N];
        for (int i = 0; i < M * N; i++) A[i] = dist(gen);
    }
    double* local_A = new double[rows_per_proc * N];
    MPI_Scatter(A, rows_per_proc * N, MPI_DOUBLE,
                local_A, rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) delete[] A;
    
    // Schritt 3: Lokale Berechnung
    double* local_y = new double[rows_per_proc];
    for (int i = 0; i < rows_per_proc; i++) {
        local_y[i] = 0.0;
        for (int j = 0; j < N; j++) {
            local_y[i] += local_A[i * N + j] * x[j];
        }
    }
    delete[] x;
    delete[] local_A;
    
    // Schritt 4: Gather y
    double* y = nullptr;
    if (rank == 0) y = new double[M];
    MPI_Gather(local_y, rows_per_proc, MPI_DOUBLE,
               y, rows_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    delete[] local_y;
    
    // Schritt 5: Ausgabe
    if (rank == 0) {
        std::cout << "First 10 results: ";
        for (int i = 0; i < 10; i++) std::cout << y[i] << " ";
        std::cout << std::endl;
        delete[] y;
    }
    
    MPI_Finalize();
    return 0;
}