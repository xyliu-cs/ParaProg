#include <iostream>
#include <chrono>

#include "utils.hpp"
// #include <openacc.h> // OpenACC Header

const float filter[3][3] = {
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
        for (int x = width + 1; x <= width * (height - 1) - 2; x++) {
            float r_sum, g_sum, b_sum;

            // contiguous memory access, maybe?    
            r_sum = buffer[(x - width - 1) * dim] * filter[0][0] + buffer[((x - width - 1) + 1) * dim] * filter[0][1] + buffer[((x - width - 1) + 2) * dim] * filter[0][2]
                     + buffer[(x - 1) * dim] * filter[1][0] + buffer[((x - 1) + 1) * dim] * filter[1][1] + buffer[((x - 1) + 2) * dim] * filter[1][2]
                     + buffer[(x + width - 1) * dim] * filter[2][0] + buffer[((x + width - 1) + 1) * dim] * filter[2][1] + buffer[((x + width - 1) + 2) * dim] * filter[2][2];
            
            g_sum = buffer[(x - width - 1) * dim + 1] * filter[0][0] + buffer[((x - width - 1) + 1) * dim + 1] * filter[0][1] + buffer[((x - width - 1) + 2) * dim + 1] * filter[0][2]
                     + buffer[(x - 1) * dim + 1] * filter[1][0] + buffer[((x - 1) + 1) * dim + 1] * filter[1][1] + buffer[((x - 1) + 2) * dim + 1] * filter[1][2]
                     + buffer[(x + width - 1) * dim + 1] * filter[2][0] + buffer[((x + width - 1) + 1) * dim + 1] * filter[2][1] + buffer[((x + width - 1) + 2) * dim + 1] * filter[2][2];

            b_sum = buffer[(x - width - 1) * dim + 2] * filter[0][0] + buffer[((x - width - 1) + 1) * dim + 2] * filter[0][1] + buffer[((x - width - 1) + 2) * dim + 2] * filter[0][2]
                     + buffer[(x - 1) * dim + 2] * filter[1][0] + buffer[((x - 1) + 1) * dim + 2] * filter[1][1] + buffer[((x - 1) + 2) * dim + 2] * filter[1][2]
                     + buffer[(x + width - 1) * dim + 2] * filter[2][0] + buffer[((x + width - 1) + 1) * dim + 2] * filter[2][1] + buffer[((x + width - 1) + 2) * dim + 2] * filter[2][2];

            
            filteredImage[x * dim] = static_cast<unsigned char>(r_sum);   // R
            filteredImage[x * dim + 1] = static_cast<unsigned char>(g_sum); // G
            filteredImage[x * dim + 2] = static_cast<unsigned char>(b_sum); // B
            
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
