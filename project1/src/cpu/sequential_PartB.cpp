//
// Created by Liu Yuxuan on 2023/9/15.
// Email: yuxuanliu1@link.cuhk.edu.cm
//
// A naive sequential implementation of image filtering
//

#include <iostream>
#include <cmath>
#include <chrono>

#include "utils.hpp"

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read input JPEG image
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    auto input_jpeg = read_from_jpeg(input_filename);
    // Apply the filter to the image
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    auto buffer = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];

    for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels; ++i)
        filteredImage[i] = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    int width = input_jpeg.width;
    int dim = input_jpeg.num_channels;

    // Nested for loop, please optimize it
    for (int y = 1; y < input_jpeg.height - 1; y++)
    {
        for (int x = 1; x < input_jpeg.width - 1; x++)
        {
                double r_sum, g_sum, b_sum;

                int r_row1_flat_base = (y - 1) * width + (x - 1);
                int red_1_1 = r_row1_flat_base * dim;
                int red_1_2 = (r_row1_flat_base + 1) * dim; 
                int red_1_3 = (r_row1_flat_base + 2) * dim;

                int r_row2_flat_base = y * width + (x - 1);
                int red_2_1 = r_row2_flat_base * dim;
                int red_2_2 = (r_row2_flat_base + 1) * dim; 
                int red_2_3 = (r_row2_flat_base + 2) * dim;

                int r_row3_flat_base = (y + 1) * width + (x - 1);
                int red_3_1 = r_row3_flat_base * dim;
                int red_3_2 = (r_row3_flat_base + 1) * dim; 
                int red_3_3 = (r_row3_flat_base + 2) * dim;

                // contiguous memory access, maybe?    
                r_sum = input_jpeg.buffer[red_1_1] * filter[0][0] + input_jpeg.buffer[red_1_2] * filter[0][1] + input_jpeg.buffer[red_1_3] * filter[0][2];
                g_sum = input_jpeg.buffer[red_1_1+1] * filter[0][0] + input_jpeg.buffer[red_1_2+1] * filter[0][1] + input_jpeg.buffer[red_1_3+1] * filter[0][2]; 
                b_sum = input_jpeg.buffer[red_1_1+2] * filter[0][0] + input_jpeg.buffer[red_1_2+2] * filter[0][1] + input_jpeg.buffer[red_1_3+2] * filter[0][2];

                r_sum += input_jpeg.buffer[red_2_1] * filter[1][0] + input_jpeg.buffer[red_2_2] * filter[1][1] + input_jpeg.buffer[red_2_3] * filter[1][2];
                g_sum += input_jpeg.buffer[red_2_1+1] * filter[1][0] + input_jpeg.buffer[red_2_2+1] * filter[1][1] + input_jpeg.buffer[red_2_3+1] * filter[1][2];
                b_sum += input_jpeg.buffer[red_2_1+2] * filter[1][0] + input_jpeg.buffer[red_2_2+2] * filter[1][1] + input_jpeg.buffer[red_2_3+2] * filter[1][2];

                r_sum += input_jpeg.buffer[red_3_1] * filter[2][0] + input_jpeg.buffer[red_3_2] * filter[2][1] + input_jpeg.buffer[red_3_3] * filter[2][2];
                g_sum += input_jpeg.buffer[red_3_1+1] * filter[2][0] + input_jpeg.buffer[red_3_2+1] * filter[2][1] + input_jpeg.buffer[red_3_3+1] * filter[2][2];
                b_sum += input_jpeg.buffer[red_3_1+2] * filter[2][0] + input_jpeg.buffer[red_3_2+2] * filter[2][1] + input_jpeg.buffer[red_3_3+2] * filter[2][2];

                int base_r = (y * width + x) * dim;
                
                filteredImage[base_r] = static_cast<unsigned char>(r_sum);   // R
                filteredImage[base_r+1] = static_cast<unsigned char>(g_sum); // G
                filteredImage[base_r+2] = static_cast<unsigned char>(b_sum); // B

        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Post-processing
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}
