#include <iostream>
#include <cmath>
#include <chrono>
#include <immintrin.h>
#include "utils.hpp"
#include <vector>

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {{1.0 / 9, 1.0 / 9, 1.0 / 9},
                                                 {1.0 / 9, 1.0 / 9, 1.0 / 9},
                                                 {1.0 / 9, 1.0 / 9, 1.0 / 9}};

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read input JPEG image
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    auto input_jpeg = read_from_jpeg(input_filename);

    // Preprocess, store reds, greens and blues separately
    auto reds = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto greens = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto blues = new unsigned char[input_jpeg.width * input_jpeg.height + 16];

    for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++)
    {
        reds[i] = input_jpeg.buffer[i * input_jpeg.num_channels];
        greens[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
        blues[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
    }

    // Mask used for shuffling when store int32s to u_int8 arrays
    __m128i shuffle = _mm_setr_epi8(0, 4, 8, 12, 
                                    -1, -1, -1, -1, 
                                    -1, -1, -1, -1, 
                                    -1, -1, -1, -1);
    // array index starts from 0!
    __m128i move_zeros = _mm_setr_epi8(0, 1, 2, 4, 
                                    5, 6, 8, 9, 
                                    10, 12, 13, 14, 
                                    3, 7, 11, 15);

    auto filteredImage = new unsigned char [input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    
    for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels;++i)
        filteredImage[i] = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Nested for loop, please optimize it
    for (int height = 1; height < input_jpeg.height - 1; height++) {
        for (int width = 1; width < input_jpeg.width - 1; width+=8) {  // Increment by 8
            __m256 sum_r = _mm256_setzero_ps();
            __m256 sum_g = _mm256_setzero_ps();
            __m256 sum_b = _mm256_setzero_ps();

            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int index = (height + i) * input_jpeg.width + (width + j);

                    // Load the 8 red chars to a 256 bits float register
                    __m128i red_chars = _mm_loadu_si128((__m128i*)(reds + index));
                    __m256i red_ints = _mm256_cvtepu8_epi32(red_chars);
                    __m256 red_floats = _mm256_cvtepi32_ps(red_ints);

                    __m128i green_chars = _mm_loadu_si128((__m128i*)(greens + index));
                    __m256i green_ints = _mm256_cvtepu8_epi32(green_chars);
                    __m256 green_floats = _mm256_cvtepi32_ps(green_ints);

                    __m128i blue_chars = _mm_loadu_si128((__m128i*)(blues + index));
                    __m256i blue_ints = _mm256_cvtepu8_epi32(blue_chars);
                    __m256 blue_floats = _mm256_cvtepi32_ps(blue_ints);

                    __m256 filter_value = _mm256_set1_ps(filter[i + 1][j + 1]);
                    
                    sum_r = _mm256_fmadd_ps(red_floats, filter_value, sum_r);
                    sum_g = _mm256_fmadd_ps(green_floats, filter_value, sum_g);
                    sum_b = _mm256_fmadd_ps(blue_floats, filter_value, sum_b);
                }
            }

            // Convert the float32 results to int32
            __m256i sum_r_ints =  _mm256_cvtps_epi32(sum_r);
            __m256i sum_g_ints =  _mm256_cvtps_epi32(sum_g);
            __m256i sum_b_ints =  _mm256_cvtps_epi32(sum_b);

            // Seperate the 256bits result to 2 128bits result
            __m128i low_r = _mm256_castsi256_si128(sum_r_ints);
            __m128i high_r = _mm256_extracti128_si256(sum_r_ints, 1);

            __m128i low_g = _mm256_castsi256_si128(sum_g_ints);
            __m128i high_g = _mm256_extracti128_si256(sum_g_ints, 1);

            __m128i low_b = _mm256_castsi256_si128(sum_b_ints);
            __m128i high_b = _mm256_extracti128_si256(sum_b_ints, 1);

            // shuffling int32s to u_int8s
            // |0|0|0|4|0|0|0|3|0|0|0|2|0|0|0|1| -> |4|3|2|1|
            __m128i trans_low_r = _mm_shuffle_epi8(low_r, shuffle);
            __m128i trans_high_r = _mm_shuffle_epi8(high_r, shuffle);

            __m128i trans_low_g = _mm_shuffle_epi8(low_g, shuffle);
            __m128i trans_high_g = _mm_shuffle_epi8(high_g, shuffle);

            __m128i trans_low_b = _mm_shuffle_epi8(low_b, shuffle);
            __m128i trans_high_b = _mm_shuffle_epi8(high_b, shuffle);

            // Unpack "low" values as RBRBRBRB
            __m128i rb_4_low = _mm_unpacklo_epi8(trans_low_r, trans_low_b);
            // Unpack low values as G0G0G0G0
            __m128i g0_4_low = _mm_unpacklo_epi8(trans_low_g, _mm_setzero_si128());

            // Unpack "high" values, which stores in the lower part now
            __m128i rb_4_hi = _mm_unpacklo_epi8(trans_high_r, trans_high_b);
            __m128i g0_4_hi = _mm_unpacklo_epi8(trans_high_g, _mm_setzero_si128());

            // interleave as RGB0RGB0RGB0
            __m128i rgb0_low = _mm_unpacklo_epi8(rb_4_low, g0_4_low); //__m128i is now full
            __m128i rgb0_hi = _mm_unpacklo_epi8(rb_4_hi, g0_4_hi); //__m128i is now full

            
            // move to 0s to the back
            __m128i rgb_low = _mm_shuffle_epi8(rgb0_low, move_zeros);
            __m128i rgb_hi = _mm_shuffle_epi8(rgb0_hi, move_zeros);

            // Store the results back to filtered image
            int base_idx = (height * input_jpeg.width + width) * input_jpeg.num_channels;
            _mm_storeu_si128((__m128i*)(&filteredImage[base_idx]), rgb_low);
            _mm_storeu_si128((__m128i*)(&filteredImage[base_idx+12]), rgb_hi); // rewrite the zeros
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height,
                         3, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Post-processing
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    delete[] reds;
    delete[] greens;
    delete[] blues;

    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
