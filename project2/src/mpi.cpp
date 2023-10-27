//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// MPI + OpenMp + SIMD + Reordering Matrix Multiplication
//

#include <mpi.h>  // MPI Header
#include <omp.h> 
#include <immintrin.h>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

#define MASTER 0

Matrix matrix_multiply_mpi(const Matrix& matrix1, const Matrix& matrix2, size_t M, size_t K, size_t N, size_t start_row, size_t end_row) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    // size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(end_row - start_row, N);
    // Your Code Here!
    // Optimizing Matrix Multiplication 
    // In addition to OpenMP, SIMD, Memory Locality and Cache Missing,
    // Further Applying MPI
    // Note:
    // You can change the argument of the function 
    // for your convenience of task division

    size_t SM = 128;  // Tile size
#pragma omp parallel for collapse(3) shared(result) 
    for (size_t i = start_row; i < end_row; i += SM) {
        for (size_t j = 0; j < N; j += SM) {
            for (size_t k = 0; k < K; k += SM) {

                for (size_t i2 = 0; i2 < SM && (i + i2) < end_row; ++i2) { 
                    int *res_row = &result[i + i2 - start_row][j];
                    const int *m1_row = &matrix1[i + i2][k];

                    for (size_t k2 = 0; k2 < SM && (k + k2) < K; ++k2) { 
                        const int *m2_row = &matrix2[k + k2][j];
                        __m256i m1_item = _mm256_set1_epi32(m1_row[k2]); // Broadcast to 8 lanes

                        for (size_t j2 = 0; j2 < SM && (j + j2) < N; j2 += 8) { // the matrix is initialized with col + 8
                            // Load 8 elements from res_row and m2_row into AVX2 registers
                            __m256i res_row_vec = _mm256_loadu_si256((__m256i*) &res_row[j2]);
                            __m256i m2_row_vec = _mm256_loadu_si256((__m256i*) &m2_row[j2]);

                            // Perform element-wise multiplication
                            __m256i v_mul = _mm256_mullo_epi32(m1_item, m2_row_vec);

                            // Add the multiplication result to the current result
                            __m256i v_res = _mm256_add_epi32(res_row_vec, v_mul);

                            // Store the result back to memory
                            _mm256_storeu_si256((__m256i*) &res_row[j2], v_res);
                        }
                    }
                }
            }
        }
    }

    return result;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 5) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable thread_num "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // What's my rank?
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    // Which node am I running on?
    int len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;

    int thread_num = atoi(argv[1]);
    omp_set_num_threads(thread_num);

    // Read Matrix
    const std::string matrix1_path = argv[2];
    const std::string matrix2_path = argv[3];
    const std::string result_path = argv[4];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);
    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);
    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    auto start_time = std::chrono::high_resolution_clock::now();

    // divide workload by row
    size_t rows_per_task = M / numtasks;
    size_t left_rows = M % numtasks;
    size_t left_assigned = 0;

    std::vector<int> cuts(numtasks + 1, 0);

    for (int i = 0; i < numtasks; i++) {
        if (i == numtasks - 1) {
            cuts[i+1] = M;
        } else if (left_assigned < left_rows) {
            cuts[i+1] = cuts[i] + rows_per_task + 1;
            left_assigned++;
        } else {
            cuts[i+1] = cuts[i] + rows_per_task;
        } 
    }
        

    Matrix result(M, N);
    // size_t start_row = taskid * rows_per_task;
    // size_t end_row = (taskid == numtasks - 1) ? M : start_row + rows_per_task;


    if (taskid == MASTER) {
        Matrix master_result = matrix_multiply_mpi(matrix1, matrix2, M, K, N, cuts[MASTER], cuts[MASTER + 1]);

        for (size_t i = cuts[MASTER]; i < cuts[MASTER + 1]; i++) {
            for (size_t j = 0; j < N; j++) {
                result[i][j] = master_result[i][j];
            }
        }

        // Directly receive results from other nodes into the result matrix
        for (int i = 1; i < numtasks; i++) {
            for(int j = cuts[i]; j < cuts[i + 1]; j++){
                MPI_Recv(result[j], N, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            }
        }        
        // Your Code Here for Synchronization!

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                  start_time);

        result.saveToFile(result_path);

        std::cout << "Output file to: " << result_path << std::endl;

        std::cout << "Multiplication Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count()
                  << " milliseconds" << std::endl;
    } else {
        Matrix local_result = matrix_multiply_mpi(matrix1, matrix2, M, K, N, cuts[taskid], cuts[taskid + 1]);

        // Your Code Here for Synchronization!
        for(int i = 0; i < (cuts[taskid + 1] - cuts[taskid]); i++){
            MPI_Send(local_result[i], N, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}