#include <iostream>
#include <vector>
#include <chrono>

#include <mpi.h>    // MPI Header

#include "utils.hpp"

#define MASTER 0
#define TAG_GATHER 0

// const int FILTER_SIZE = 3;
// const double filter[FILTER_SIZE][FILTER_SIZE] = {
//     {1.0 / 9, 1.0 / 9, 1.0 / 9},
//     {1.0 / 9, 1.0 / 9, 1.0 / 9},
//     {1.0 / 9, 1.0 / 9, 1.0 / 9}
// };

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numprocess;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocess);
    // What's my rank?
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    // Which node am I running on?
    int len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;
    MPI_Request *requests = new MPI_Request[numprocess - 1];
    
    // Read JPEG File
    const char * input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }



    // Divide the task
    // For example, there are 11 pixels and 3 tasks, 
    // we try to divide to 4 4 3 instead of 3 3 5
    int total_pixel_num = input_jpeg.width * (input_jpeg.height - 2) - 2; // little trick
    int pixel_num_per_task = total_pixel_num / numprocess;
    int left_pixel_num = total_pixel_num % numprocess;

    std::vector<int> cuts(numprocess + 1, 0);
    cuts[0] = input_jpeg.width + 1;
    int left_pixel_assigned = 0;

    // not continous indices
    for (int i = 0; i < numprocess; i++) {
        if (left_pixel_assigned < left_pixel_num) {
            cuts[i+1] = cuts[i] + pixel_num_per_task + 1;
            left_pixel_assigned++;
        } else cuts[i+1] = cuts[i] + pixel_num_per_task;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    if (taskid == MASTER) {
        int width = input_jpeg.width;
        int height = input_jpeg.height;
        int channels = input_jpeg.num_channels;
        auto filteredImage = new unsigned char[width * height * channels]();

        double filter[3][3] = {
            {1.0 / 9, 1.0 / 9, 1.0 / 9},
            {1.0 / 9, 1.0 / 9, 1.0 / 9},
            {1.0 / 9, 1.0 / 9, 1.0 / 9}
        };

        for (int i = cuts[MASTER]; i < cuts[MASTER + 1]; i++) {
            if ((i+1) % width == 0 || i % width == 0) { // boundary points
                continue;
            } 

            double r_sum = 0, g_sum = 0, b_sum = 0;
            
            int r_row1_flat_base = i - 1 * width - 1;
            int red_1_1 = r_row1_flat_base * channels;
            int red_1_2 = (r_row1_flat_base + 1) * channels; 
            int red_1_3 = (r_row1_flat_base + 2) * channels;

            int r_row2_flat_base = i - 1;
            int red_2_1 = r_row2_flat_base * channels;
            int red_2_2 = (r_row2_flat_base + 1) * channels; 
            int red_2_3 = (r_row2_flat_base + 2) * channels;

            int r_row3_flat_base = i + 1 * width - 1;
            int red_3_1 = r_row3_flat_base * channels;
            int red_3_2 = (r_row3_flat_base + 1) * channels; 
            int red_3_3 = (r_row3_flat_base + 2) * channels;

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
              
            int base_idx = i * channels;

            filteredImage[base_idx] = static_cast<unsigned char>(r_sum);
            filteredImage[base_idx+1] = static_cast<unsigned char>(g_sum);
            filteredImage[base_idx+2] = static_cast<unsigned char>(b_sum);
        }

        // Receive the filtered contents from each slave executors
        for (int i = MASTER + 1; i < numprocess; i++) {
            unsigned char* start_pos = filteredImage + (cuts[i] * channels);
            int length = (cuts[i+1] - cuts[i]) * channels;
            MPI_Irecv(start_pos, length, MPI_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &requests[i - MASTER - 1]);
        }
        MPI_Waitall(numprocess - 1, requests, MPI_STATUSES_IGNORE);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);


        // Save the image
        const char* output_filepath = argv[2];
        std::cout << "Output file to: " << output_filepath << "\n";
        JPEGMeta output_jpeg{filteredImage, width, height, channels, input_jpeg.color_space};
        if (write_to_jpeg(output_jpeg, output_filepath)) {
            std::cerr << "Failed to write output JPEG to file\n";
            MPI_Finalize();
            return -1;
        }

        // Release the memory
        delete[] requests;
        delete[] input_jpeg.buffer;
        delete[] filteredImage;
        std::cout << "Transformation Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    }

    // The tasks for the slave executor
    // 1. Filter the RGB contents 
    // 2. Send the filtered contents back to the master executor
    else {
        int width = input_jpeg.width;
        int channels = input_jpeg.num_channels;
        int length = (cuts[taskid + 1] - cuts[taskid]) * channels; 
        auto filteredImage_frag = new unsigned char[length]();

        double filter[3][3] = {
            {1.0 / 9, 1.0 / 9, 1.0 / 9},
            {1.0 / 9, 1.0 / 9, 1.0 / 9},
            {1.0 / 9, 1.0 / 9, 1.0 / 9}
        };
        
        for (int i = cuts[taskid]; i < cuts[taskid + 1]; i++) {
            if ((i+1) % width == 0 || i % width == 0) { // boundary points
                continue;
            }

            double r_sum = 0, g_sum = 0, b_sum = 0;
            
            int r_row1_flat_base = i - 1 * width - 1;
            int red_1_1 = r_row1_flat_base * channels;
            int red_1_2 = (r_row1_flat_base + 1) * channels; 
            int red_1_3 = (r_row1_flat_base + 2) * channels;

            int r_row2_flat_base = i - 1;
            int red_2_1 = r_row2_flat_base * channels;
            int red_2_2 = (r_row2_flat_base + 1) * channels; 
            int red_2_3 = (r_row2_flat_base + 2) * channels;

            int r_row3_flat_base = i + 1 * width - 1;
            int red_3_1 = r_row3_flat_base * channels;
            int red_3_2 = (r_row3_flat_base + 1) * channels; 
            int red_3_3 = (r_row3_flat_base + 2) * channels;

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
            
            int j = (i - cuts[taskid]) * channels;

            filteredImage_frag[j] = static_cast<unsigned char>(r_sum);
            filteredImage_frag[j+1] = static_cast<unsigned char>(g_sum);
            filteredImage_frag[j+2] = static_cast<unsigned char>(b_sum);
        }

        // Send the gray image back to the master
        MPI_Request send_request;
        MPI_Isend(filteredImage_frag, length, MPI_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD, &send_request);
        MPI_Wait(&send_request, MPI_STATUS_IGNORE);
        
        // Release the memory
        delete[] filteredImage_frag;
    }

    MPI_Finalize();
    return 0;
}
