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
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
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
    
    // Separate R, G, B channels into three continuous arrays
    auto rChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto gChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto bChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    
    for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++) {
        rChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels];
        gChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
        bChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
    }

    // Transforming the R, G, B channels in parallel
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels]();
    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for default(none) shared(rChannel, gChannel, bChannel, filteredImage, input_jpeg, filter)
    for (int y = 1; y <= input_jpeg.height - 2; y++) {
        for (int x = 1; x <= input_jpeg.width -2; x++) {

            int width = input_jpeg.width;
            float r_sum = 0.0, g_sum = 0.0, b_sum = 0.0;

            for (int dy = -1; dy <= 1; dy++) {
                int row_1st = (y+dy) * width + x - 1;
                int row_2nd = (y+dy) * width + x ; 
                int row_3nd = (y+dy) * width + x + 1;
                
                r_sum += rChannel[row_1st] * filter[1+dy][0] + rChannel[row_2nd] * filter[1+dy][1] + rChannel[row_3nd] * filter[1+dy][2];
                g_sum += gChannel[row_1st] * filter[1+dy][0] + gChannel[row_2nd] * filter[1+dy][1] + gChannel[row_3nd] * filter[1+dy][2];
                b_sum += bChannel[row_1st] * filter[1+dy][0] + bChannel[row_2nd] * filter[1+dy][1] + bChannel[row_3nd] * filter[1+dy][2];
            }

            int base_r = (y * width + x) * input_jpeg.num_channels;
            
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
    delete[] rChannel;
    delete[] gChannel;
    delete[] bChannel;
    delete[] filteredImage;
    
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}
