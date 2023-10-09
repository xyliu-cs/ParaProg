#include <iostream>
#include <vector>
#include <chrono>

#include <mpi.h>    // MPI Header

#include "utils.hpp"

#define MASTER 0
#define TAG_GATHER 0

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
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
    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numprocess, taskid;

    MPI_Comm_size(MPI_COMM_WORLD, &numprocess);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

    MPI_Status status;
    MPI_Request *requests = new MPI_Request[numprocess - 1];
    
    // Read JPEG File
    const char * input_filepath = argv[1];
    if (taskid == MASTER) {
        std::cout << "Input file from: " << input_filepath << "\n";
    }

    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int dim = input_jpeg.num_channels;
    auto filteredImage = new unsigned char[width * height * dim]();


    // Divide the task
    int workload = ((height - 2) + numprocess - 1) / numprocess;
    int start_row = taskid * workload + 1;
    int end_row = (taskid == numprocess - 1) ? (height - 1) : ((taskid + 1) * workload + 1);


    auto start_time = std::chrono::high_resolution_clock::now();

    for (int y = start_row; y < end_row; y++) {
        for (int x = 1; x < width - 1; x++) {
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
    MPI_Barrier(MPI_COMM_WORLD); // a little sync here
    if (taskid == MASTER) {
        // Receive the filtered contents from each slave executors
        for (int i = MASTER + 1; i < numprocess; i++) {
            int workload = ((height - 2) + numprocess - 1) / numprocess;
            int start_row = i * workload + 1;
            unsigned char* start_pos = filteredImage + start_row * width * dim;
            int end_row = (i == numprocess - 1) ? (height - 1) : ((i + 1) * workload + 1);
            int length = (end_row - start_row) * width * dim;
            // MPI_Recv(start_pos, length, MPI_UNSIGNED_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Irecv(start_pos, length, MPI_UNSIGNED_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &requests[i - MASTER - 1]);
        }
        MPI_Waitall(numprocess - 1, requests, MPI_STATUSES_IGNORE);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Save the image
        const char* output_filepath = argv[2];
        std::cout << "Output file to: " << output_filepath << "\n";
        JPEGMeta output_jpeg{filteredImage, width, height, dim, input_jpeg.color_space};
        if (write_to_jpeg(output_jpeg, output_filepath)) {
            std::cerr << "Failed to write output JPEG to file\n";
            MPI_Finalize();
            return -1;
        }
        std::cout << "Transformation Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";        
    } 
    
    else { // slave processes
        MPI_Request send_request;
        int length =  (end_row - start_row) * width * dim;
        unsigned char* start_pos = filteredImage + start_row * width * dim;
        // MPI_Send(start_pos, length, MPI_UNSIGNED_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);

        MPI_Isend(start_pos, length, MPI_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD, &send_request);
        MPI_Wait(&send_request, MPI_STATUS_IGNORE);
    }
    // Release the memory
    delete[] requests;
    delete[] input_jpeg.buffer;
    delete[] filteredImage;

    MPI_Finalize();
    return 0;
}