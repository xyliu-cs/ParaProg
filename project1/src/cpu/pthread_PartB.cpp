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
    int w = data->width;
    int dim = 3;

    for (int row_idx = data->start_row; row_idx <= data->end_row; row_idx++) {
        for (int x = 1; x < w - 1; x++) {
            float r_sum;
            float g_sum;
            float b_sum;

            int r_row1_flat_base = (row_idx - 1) * w + (x - 1);
            int red_1_1 = r_row1_flat_base * dim;
            int red_1_2 = (r_row1_flat_base + 1) * dim; 
            int red_1_3 = (r_row1_flat_base + 2) * dim;

            int r_row2_flat_base = row_idx * w + (x - 1);
            int red_2_1 = r_row2_flat_base * dim;
            int red_2_2 = (r_row2_flat_base + 1) * dim; 
            int red_2_3 = (r_row2_flat_base + 2) * dim;

            int r_row3_flat_base = (row_idx + 1) * w + (x - 1);
            int red_3_1 = r_row3_flat_base * dim;
            int red_3_2 = (r_row3_flat_base + 1) * dim; 
            int red_3_3 = (r_row3_flat_base + 2) * dim;

            // contiguous memory access, maybe?    
            r_sum = input[red_1_1] * filter[0][0] + input[red_1_2] * filter[0][1] + input[red_1_3] * filter[0][2];
            g_sum = input[red_1_1+1] * filter[0][0] + input[red_1_2+1] * filter[0][1] + input[red_1_3+1] * filter[0][2]; 
            b_sum = input[red_1_1+2] * filter[0][0] + input[red_1_2+2] * filter[0][1] + input[red_1_3+2] * filter[0][2];

            r_sum += input[red_2_1] * filter[1][0] + input[red_2_2] * filter[1][1] + input[red_2_3] * filter[1][2];
            g_sum += input[red_2_1+1] * filter[1][0] + input[red_2_2+1] * filter[1][1] + input[red_2_3+1] * filter[1][2];
            b_sum += input[red_2_1+2] * filter[1][0] + input[red_2_2+2] * filter[1][1] + input[red_2_3+2] * filter[1][2];

            r_sum += input[red_3_1] * filter[2][0] + input[red_3_2] * filter[2][1] + input[red_3_3] * filter[2][2];
            g_sum += input[red_3_1+1] * filter[2][0] + input[red_3_2+1] * filter[2][1] + input[red_3_3+1] * filter[2][2];
            b_sum += input[red_3_1+2] * filter[2][0] + input[red_3_2+2] * filter[2][1] + input[red_3_3+2] * filter[2][2];

            int out_idx_r = (row_idx * w + x) * 3;;

            output[out_idx_r] = static_cast<unsigned char>(r_sum);
            output[out_idx_r + 1] = static_cast<unsigned char>(g_sum);
            output[out_idx_r + 2] = static_cast<unsigned char>(b_sum);
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
            int dim = 3;
            double r_sum = 0, g_sum = 0, b_sum = 0;
            
            int r_row1_flat_base = (height - 1) * w + (width - 1);
            int red_1_1 = r_row1_flat_base * dim;
            int red_1_2 = (r_row1_flat_base + 1) * dim; 
            int red_1_3 = (r_row1_flat_base + 2) * dim;

            int r_row2_flat_base = height * w + (width - 1);
            int red_2_1 = r_row2_flat_base * dim;
            int red_2_2 = (r_row2_flat_base + 1) * dim; 
            int red_2_3 = (r_row2_flat_base + 2) * dim;

            int r_row3_flat_base = (height + 1) * w + (width - 1);
            int red_3_1 = r_row3_flat_base * dim;
            int red_3_2 = (r_row3_flat_base + 1) * dim; 
            int red_3_3 = (r_row3_flat_base + 2) * dim;

            // contiguous memory access, maybe?    
            r_sum = input[red_1_1] * filter[0][0] + input[red_1_2] * filter[0][1] + input[red_1_3] * filter[0][2];
            g_sum = input[red_1_1+1] * filter[0][0] + input[red_1_2+1] * filter[0][1] + input[red_1_3+1] * filter[0][2]; 
            b_sum = input[red_1_1+2] * filter[0][0] + input[red_1_2+2] * filter[0][1] + input[red_1_3+2] * filter[0][2];

            r_sum += input[red_2_1] * filter[1][0] + input[red_2_2] * filter[1][1] + input[red_2_3] * filter[1][2];
            g_sum += input[red_2_1+1] * filter[1][0] + input[red_2_2+1] * filter[1][1] + input[red_2_3+1] * filter[1][2];
            b_sum += input[red_2_1+2] * filter[1][0] + input[red_2_2+2] * filter[1][1] + input[red_2_3+2] * filter[1][2];

            r_sum += input[red_3_1] * filter[2][0] + input[red_3_2] * filter[2][1] + input[red_3_3] * filter[2][2];
            g_sum += input[red_3_1+1] * filter[2][0] + input[red_3_2+1] * filter[2][1] + input[red_3_3+1] * filter[2][2];
            b_sum += input[red_3_1+2] * filter[2][0] + input[red_3_2+2] * filter[2][1] + input[red_3_3+2] * filter[2][2];

            
            int base = (height * w + width) * 3;

            output[base] = static_cast<unsigned char>(r_sum);
            output[base + 1] = static_cast<unsigned char>(g_sum);
            output[base + 2] = static_cast<unsigned char>(b_sum);
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

    // Write filtered image to output JPEG
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
