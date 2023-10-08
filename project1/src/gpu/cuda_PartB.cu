#include <iostream>
#include <cuda_runtime.h> // CUDA Header
#include "utils.hpp"

__constant__ float filter[3][3] = {
    { 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f },
    { 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f },
    { 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f }
};

// CUDA kernel functon：RGB filtering
__global__ void matrixFilter(const unsigned char* input, unsigned char* output, int width, int height, int dim)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    if (x >= 1 && x <= (width-2) && y >= 1 && y <= (height-2)) {
        float r_sum = 0.0, g_sum = 0.0, b_sum = 0.0;
        for (int dy = -1; dy <= 1; dy++) {

            int base_1 = ((y+dy) * width + x-1) * dim;
            int base_2 = ((y+dy) * width + x) * dim;
            int base_3 = ((y+dy) * width + x+1) * dim;
            
            r_sum += input[base_1] * filter[1+dy][0] + input[base_2] * filter[1+dy][1] + input[base_3] * filter[1+dy][2];
            g_sum += input[base_1 + 1] * filter[1+dy][0] + input[base_2 + 1] * filter[1+dy][1] + input[base_3 + 1] * filter[1+dy][2];
            b_sum += input[base_1 + 2] * filter[1+dy][0] + input[base_2 + 2] * filter[1+dy][1] + input[base_3 + 2] * filter[1+dy][2];

        }

        output[(y * width + x) * dim] = static_cast<unsigned char>(r_sum);
        output[(y * width + x) * dim + 1] = static_cast<unsigned char>(g_sum);
        output[(y * width + x) * dim + 2] = static_cast<unsigned char>(b_sum);

    }
}


int main(int argc, char** argv)
{
    // Verify input argument format
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read from input JPEG
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    // Allocate memory on host (CPU)
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels]();   // num_channels = 3
    
    // Allocate memory on device (GPU)
    unsigned char* d_input;
    unsigned char* d_output;
    cudaMalloc((void**)&d_input, 
                input_jpeg.width * input_jpeg.height * input_jpeg.num_channels * sizeof(unsigned char));
    cudaMalloc((void**)&d_output,
               input_jpeg.width * input_jpeg.height * input_jpeg.num_channels * sizeof(unsigned char));
    
    // Copy input data from host to device
    cudaMemcpy(d_input, input_jpeg.buffer,
               input_jpeg.width * input_jpeg.height * input_jpeg.num_channels *
                   sizeof(unsigned char),
               cudaMemcpyHostToDevice);
    
    // Computation: RGB filtering
    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blockSize(16, 16);  // For example, 16x16 threads per block
    dim3 numBlocks((input_jpeg.width + blockSize.x - 1) / blockSize.x,
               (input_jpeg.height + blockSize.y - 1) / blockSize.y);

    // std::cout << "Width: " << input_jpeg.width << " Height " << input_jpeg.height << std::endl;
    // std::cout << "Block Size: " << blockSize.x << "x" << blockSize.y << std::endl;
    // std::cout << "Grid Size: " << numBlocks.x << "x" << numBlocks.y << std::endl;

    cudaEventRecord(start, 0); // GPU start time
    matrixFilter<<<numBlocks, blockSize>>>(d_input, d_output, input_jpeg.width,
                                        input_jpeg.height,
                                        input_jpeg.num_channels);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0); // GPU end time
    cudaEventSynchronize(stop);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess)
    //     printf("Error: %s\n", cudaGetErrorString(err));
    
    // Print the result of the GPU computation
    cudaEventElapsedTime(&gpuDuration, start, stop);
    
    // Copy output data from device to host
    cudaMemcpy(filteredImage, d_output,
               input_jpeg.width * input_jpeg.height * input_jpeg.num_channels * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);

    // Write filtered Image to output JPEG
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels,
                         input_jpeg.color_space};                    
    if (write_to_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Release allocated memory on device and host
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}