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
                              buffer[0 : width * height * dim])

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma acc parallel present(filteredImage[0 : width * height * dim],             \
                             buffer[0 : width * height * dim]) num_gangs(1024)
    {
        #pragma acc loop independent
        for (int y = 1; y <= height - 2; y++) {
        #pragma acc loop independent
            for (int x = 1; x <= width -2; x++) {
                
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
                r_sum = buffer[red_1_1] * filter[0][0] + buffer[red_1_2] * filter[0][1] + buffer[red_1_3] * filter[0][2];
                g_sum = buffer[red_1_1+1] * filter[0][0] + buffer[red_1_2+1] * filter[0][1] + buffer[red_1_3+1] * filter[0][2]; 
                b_sum = buffer[red_1_1+2] * filter[0][0] + buffer[red_1_2+2] * filter[0][1] + buffer[red_1_3+2] * filter[0][2];

                r_sum += buffer[red_2_1] * filter[1][0] + buffer[red_2_2] * filter[1][1] + buffer[red_2_3] * filter[1][2];
                g_sum += buffer[red_2_1+1] * filter[1][0] + buffer[red_2_2+1] * filter[1][1] + buffer[red_2_3+1] * filter[1][2];
                b_sum += buffer[red_2_1+2] * filter[1][0] + buffer[red_2_2+2] * filter[1][1] + buffer[red_2_3+2] * filter[1][2];

                r_sum += buffer[red_3_1] * filter[2][0] + buffer[red_3_2] * filter[2][1] + buffer[red_3_3] * filter[2][2];
                g_sum += buffer[red_3_1+1] * filter[2][0] + buffer[red_3_2+1] * filter[2][1] + buffer[red_3_3+1] * filter[2][2];
                b_sum += buffer[red_3_1+2] * filter[2][0] + buffer[red_3_2+2] * filter[2][1] + buffer[red_3_3+2] * filter[2][2];

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
