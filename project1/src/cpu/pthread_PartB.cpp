//
// Created by Zhang Na on 2023/9/15.
// Email: nazhang@link.cuhk.edu.cn
//
// Pthread implementation of transforming a JPEG image from RGB to gray
//

#include <iostream>
#include <chrono>
#include <pthread.h>
#include <cmath>
#include "utils.hpp"

// Structure to pass data to each thread
struct ThreadData {
    unsigned char* input_buffer;
    unsigned char* output_buffer;
    int start_row;
    int end_row;
    int width;
    int height;
};


const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};


void* rgbFilter(void* arg) {
    ThreadData* data = reinterpret_cast<ThreadData*>(arg);
    unsigned char *input = data->input_buffer;
    unsigned char *output = data->output_buffer;
    int width = data->width;

    for (int row_idx = data->start_row; row_idx <= data->end_row; row_idx++) {
        for (int x = 1; x < width - 1; x++) {
            float filtered_sum_r = 0;
            float filtered_sum_g = 0;
            float filtered_sum_b = 0;

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int r_idx = ((row_idx + dy) * width + (x + dx)) * 3;
                    int g_idx = r_idx + 1;
                    int b_idx = g_idx + 1;

                    double current_filter = filter[dy+1][dx+1];

                    filtered_sum_r += input[r_idx] * current_filter;
                    filtered_sum_g += input[g_idx] * current_filter;
                    filtered_sum_b += input[b_idx] * current_filter;
                }
            }

            int out_idx_r = (row_idx * width + x) * 3;
            int out_idx_g = out_idx_r + 1;
            int out_idx_b = out_idx_g + 1;

            output[out_idx_r] = static_cast<unsigned char>(std::round(filtered_sum_r));
            output[out_idx_g] = static_cast<unsigned char>(std::round(filtered_sum_g));
            output[out_idx_b] = static_cast<unsigned char>(std::round(filtered_sum_b));
        }
    }

    return nullptr;
}


void* rgbFilter_seq(void* arg) {
    ThreadData* data = reinterpret_cast<ThreadData*>(arg);
    unsigned char *input = data->input_buffer;
    unsigned char *output = data->output_buffer;
    int w = data->width;
    int h = data->height;


    for (int height = 1; height < h - 1; height++)
    {
        for (int width = 1; width < w - 1; width++)
        {
            int sum_r = 0, sum_g = 0, sum_b = 0;
            for (int i = -1; i <= 1; i++)
            {
                for (int j = -1; j <= 1; j++)
                {
                    int channel_value_r = input[((height + i) * w + (width + j)) * 3];
                    int channel_value_g = input[((height + i) * w + (width + j)) * 3 + 1];
                    int channel_value_b = input[((height + i) * w + (width + j)) * 3 + 2];
                    sum_r += channel_value_r * filter[i + 1][j + 1];
                    sum_g += channel_value_g * filter[i + 1][j + 1];
                    sum_b += channel_value_b * filter[i + 1][j + 1];
                }
            }
            output[(height * w + width) * 3]
                = static_cast<unsigned char>(std::round(sum_r));
            output[(height * w + width) * 3 + 1]
                = static_cast<unsigned char>(std::round(sum_g));
            output[(height * w + width) * 3 + 2]
                = static_cast<unsigned char>(std::round(sum_b));
        }
    }    
}



int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg num_threads\n";
        return -1;
    }

    int num_threads = std::stoi(argv[3]); // User-specified thread count

    // Read from input JPEG
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";


    auto input_jpeg = read_from_jpeg(input_filepath);
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    auto start_time = std::chrono::high_resolution_clock::now();

    if (num_threads == 1) {
        thread_data[0].input_buffer = input_jpeg.buffer;
        thread_data[0].output_buffer = filteredImage;
        thread_data[0].width = input_jpeg.width;
        thread_data[0].height = input_jpeg.height;
        pthread_create(&threads[0], nullptr, rgbFilter_seq, &thread_data[0]);
    }

    else {
        int row_split = (input_jpeg.height - 2) / num_threads;
        for (int i = 0; i < num_threads; i++) {
            thread_data[i].input_buffer = input_jpeg.buffer;
            thread_data[i].output_buffer = filteredImage;
            thread_data[i].width = input_jpeg.width;
            thread_data[i].start_row = 1 + i * row_split; // start from the 2nd row
            thread_data[i].end_row = (i == num_threads - 1) ? (input_jpeg.height - 2) : ((i + 1) * row_split);
            pthread_create(&threads[i], nullptr, rgbFilter, &thread_data[i]);
        }
    }
    
    // Wait for all threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Write GrayImage to output JPEG
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, 3, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Release allocated memory
    delete[] input_jpeg.buffer;
    delete[] filteredImage;


    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";

    return 0;
}
