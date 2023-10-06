//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// OpenACC implementation of transforming a JPEG image from RGB to gray
//

#include <iostream>
#include <chrono>

#include "utils.hpp"
// #include <openacc.h> // OpenACC Header

const double filter[3][3] = {
            {1.0 / 9, 1.0 / 9, 1.0 / 9},
            {1.0 / 9, 1.0 / 9, 1.0 / 9},
            {1.0 / 9, 1.0 / 9, 1.0 / 9}
        };

int main(int argc, char **argv)
{
    // Verify input argument format
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read from input JPEG
    const char *input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    JPEGMeta input_jpeg = read_from_jpeg(input_filepath);

    // Computation: RGB to Gray
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int dim = input_jpeg.num_channels;

    unsigned char *filteredImage = new unsigned char[width * height * dim]();
    unsigned char *buffer = new unsigned char[width * height * dim]();

    for (int i = 0; i < width * height * dim; i++)
    {
        buffer[i] = input_jpeg.buffer[i];
    }

    #pragma acc enter data copyin(filteredImage[0 : width * height * dim], \
                              buffer[0 : width * height * dim], filter)

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma acc parallel present(filteredImage[0 : width * height * dim],             \
                             buffer[0 : width * height * dim], filter) num_gangs(1024)
    {
        #pragma acc loop independent
        for (int y = 1; y <= height - 2; y++) {
            for (int x = 1; x <= width -2; x++) {

                double r_sum = 0.0, g_sum = 0.0, b_sum = 0.0;

                for (int dy = -1; dy <= 1; dy++) {
                    int row_1st = ((y+dy) * width + x - 1) * dim;
                    int row_2nd = ((y+dy) * width + x) * dim ; 
                    int row_3nd = ((y+dy) * width + x + 1) * dim;
                    
                    r_sum += buffer[row_1st] * filter[1+dy][0] + buffer[row_2nd] * filter[1+dy][1] + buffer[row_3nd] * filter[1+dy][2];
                    g_sum += buffer[row_1st+1] * filter[1+dy][0] + buffer[row_2nd+1] * filter[1+dy][1] + buffer[row_3nd+1] * filter[1+dy][2];
                    b_sum += buffer[row_1st+2] * filter[1+dy][0] + buffer[row_2nd+2] * filter[1+dy][1] + buffer[row_3nd+2] * filter[1+dy][2];
                }

                int base_r = (y * width + x) * dim;
                
                filteredImage[base_r] = static_cast<unsigned char>(r_sum);   // R
                filteredImage[base_r+1] = static_cast<unsigned char>(g_sum); // G
                filteredImage[base_r+2] = static_cast<unsigned char>(b_sum); // B

            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    #pragma acc exit data copyout(filteredImage[0 : width * height * dim], buffer[0 : width * height * dim])

    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // Write GrayImage to output JPEG
    const char *output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels,
                         input_jpeg.color_space};

    if (write_to_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Release allocated memory
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    delete[] buffer;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
