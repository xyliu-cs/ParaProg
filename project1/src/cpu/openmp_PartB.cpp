#include <iostream>
#include <chrono>
#include <omp.h>    // OpenMP header
#include "utils.hpp"



const double filter[3][3] = {
            {1.0 / 9, 1.0 / 9, 1.0 / 9},
            {1.0 / 9, 1.0 / 9, 1.0 / 9},
            {1.0 / 9, 1.0 / 9, 1.0 / 9}
        };

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg num_threads\n";
        return -1;
    }
    // Read input JPEG image
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    int num_threads_defined = std::stoi(argv[3]); // User-specified thread count
    
    // Separate R, G, B channels into three continuous arrays
    // auto rChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    // auto gChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    // auto bChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    
    // for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++) {
    //     rChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels];
    //     gChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
    //     bChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
    // }

    // Transforming the R, G, B channels in parallel
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels]();
    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads(num_threads_defined) default(none) shared(filteredImage, input_jpeg, filter)
    for (int y = 1; y <= input_jpeg.height - 2; y++) {
        for (int x = 1; x <= input_jpeg.width - 2; x++) {

            int width = input_jpeg.width;
            int dim = input_jpeg.num_channels;

            float r_sum = 0.0, g_sum = 0.0, b_sum = 0.0;

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
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Save output JPEG filtered image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to save output JPEG image\n";
        return -1;
    }

    // Release the allocated memory
    delete[] input_jpeg.buffer;
    // delete[] rChannel;
    // delete[] gChannel;
    // delete[] bChannel;
    delete[] filteredImage;
    
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}
