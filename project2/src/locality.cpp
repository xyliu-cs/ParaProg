//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Reordering Matrix Multiplication
//

#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

Matrix matrix_multiply_locality(const Matrix& matrix1, const Matrix& matrix2) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }
    // A = M x K, B = K x N --> C = M x N
    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N);

    // Your Code Here!
    // Optimizing Matrix Multiplication 
    // Considering Memory Locality and Avoiding Cache Missing
    // Hints:
    // 1. Change the order of the tripple nested loop
    // 2. Apply Tiled Matrix Multiplication

    size_t SM = 128; // Size of the tile

    // free to unroll
    if (N % 2 == 0)
    {
    for (size_t i = 0; i < M; i += SM) {
        for (size_t j = 0; j < N; j += SM) {
            for (size_t k = 0; k < K; k += SM) {

                for (size_t ii = 0; ii < SM && (i + ii) < M; ++ii) {
                    int *res_tiled_row = &result[i + ii][j];
                    const int *m1_tiled_row = &matrix1[i + ii][k];

                    for (size_t kk = 0; kk < SM && (k + kk) < K; ++kk) {
                        const int *m2_tiled_row = &matrix2[k + kk][j];
                        // int m1_item = rmul1[k2];

                        for (size_t jj = 0; jj < SM && (j + jj) < N; jj+=2) { 
                            res_tiled_row[jj] += m1_tiled_row[kk] * m2_tiled_row[jj];
                            res_tiled_row[jj+1] += m1_tiled_row[kk] * m2_tiled_row[jj+1];
                            // res_tiled_row[jj+2] += m1_tiled_row[kk] * m2_tiled_row[jj+2];
                            // res_tiled_row[jj+3] += m1_tiled_row[kk] * m2_tiled_row[jj+3];

                        }
                    }
                }
            }
        }
    }
    }
    // is it a good remedy? probably too naive
    else 
    {
        for (size_t i = 0; i < M; i += SM) {
            for (size_t j = 0; j < N - 1; j += SM) {
                for (size_t k = 0; k < K; k += SM) {

                    for (size_t ii = 0; ii < SM && (i + ii) < M; ++ii) {
                        int *res_tiled_row = &result[i + ii][j];
                        const int *m1_tiled_row = &matrix1[i + ii][k];

                        for (size_t kk = 0; kk < SM && (k + kk) < K; ++kk) {
                            const int *m2_tiled_row = &matrix2[k + kk][j];
                            // int m1_item = rmul1[k2];
                            
                            for (size_t jj = 0; jj < SM && (j + jj) < N; jj+=2) { 
                                res_tiled_row[jj] += m1_tiled_row[kk] * m2_tiled_row[jj];
                                res_tiled_row[jj+1] += m1_tiled_row[kk] * m2_tiled_row[jj+1];
                                // res_tiled_row[jj+2] += m1_tiled_row[kk] * m2_tiled_row[jj+2];
                                // res_tiled_row[jj+3] += m1_tiled_row[kk] * m2_tiled_row[jj+3];

                            }
                        }
                    }
                }
            }
        }
        // j = N - 1, naive
        for (size_t i = 0; i < M; i++) {
            int row_sum = 0;
            for (size_t k = 0; k < K; k ++) {
                row_sum += matrix1[i][k] * matrix2[k][N-1];
            }
            result[i][N-1] = row_sum;
        }
    }
    
    return result;
}


int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    const std::string matrix1_path = argv[1];

    const std::string matrix2_path = argv[2];

    const std::string result_path = argv[3];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_locality(matrix1, matrix2);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    result.saveToFile(result_path);

    std::cout << "Output file to: " << result_path << std::endl;

    std::cout << "Multiplication Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    return 0;
}