//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// OpenMp + SIMD + Reordering Matrix Multiplication
//scan

#include <immintrin.h>
#include <omp.h> 
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

Matrix matrix_multiply_openmp(const Matrix& matrix1, const Matrix& matrix2) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N);

    size_t SM = 128; // Size of the tile
#pragma omp parallel for collapse(3) shared(result) 
    for (size_t i = 0; i < M; i += SM) {
        for (size_t j = 0; j < N; j += SM) {
            for (size_t k = 0; k < K; k += SM) {

                for (size_t i2 = 0; i2 < SM && (i + i2) < M; ++i2) { 
                    int *res_row = &result[i + i2][j];
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
            "Invalid argument, should be: ./executable thread_num"
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    int thread_num = atoi(argv[1]);
    omp_set_num_threads(thread_num);

    const std::string matrix1_path = argv[2];

    const std::string matrix2_path = argv[3];

    const std::string result_path = argv[4];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_openmp(matrix1, matrix2);

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