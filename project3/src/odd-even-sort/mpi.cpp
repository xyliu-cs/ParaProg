//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Odd-Even Sort with MPI
//

#include <iostream>
#include <vector>
#include <mpi.h>
#include "../utils.hpp"

#define MASTER 0


void localOddEvenSort(std::vector<int>& vec, int start, int end) {
    bool sorted = false;

    while (!sorted) {
        sorted = true;

        // Perform the odd phase
        for (int i = start + 1; i <= (end - 1); i += 2) {
            if (vec[i] > vec[i + 1]) {
                std::swap(vec[i], vec[i + 1]);
                sorted = false;
            }
        }

        // Perform the even phase
        for (int i = start; i <= (end - 1); i += 2) {
            if (vec[i] > vec[i + 1]) {
                std::swap(vec[i], vec[i + 1]);
                sorted = false;
            }
        }
    }
    // std::cout << "[local Sort] The Sorted vector elements are : "<< std::endl;
    // for(int i=start; i <= end; i++)
    //     std::cout << vec.at(i) << ' ';
}



void oddEvenSort(std::vector<int>& vec, int numtasks, int taskid, MPI_Status* status) {
    int n = vec.size();
    int extra = n % numtasks;
    int load = n / numtasks;
    int local_n = load + (taskid < extra ? 1 : 0);
    int start_index = taskid * load + std::min(taskid, extra);
    int end_index = start_index + local_n - 1;  // inclusive

    // printf("My task ID is %d, my start index is %d, my end index is %d \n", taskid, start_index, end_index);

    // Local sort
    localOddEvenSort(vec, start_index, end_index);

    if (numtasks == 1) 
        return;

    // alternating Odd-Even sort
    for (int phase = 0; phase < n; phase++) {
        // if (taskid == MASTER) {
        //     printf("The phase is: %d\n", phase);
        // }

        bool is_local_sorted = true;
        

        // Even phase
        if (phase % 2 == 0) {

            // even-indexed tasks
            if (taskid % 2 == 0 && taskid != numtasks - 1) { 
                int next_val;
                // printf("My task ID is %d, expect to Send and Receive from the next task \n", taskid);

                MPI_Sendrecv(&vec[end_index], 1, MPI_INT, taskid + 1, 0,
                            &next_val, 1, MPI_INT, taskid + 1, 0,
                            MPI_COMM_WORLD, status);
                // printf("My task ID is %d, Successfully Send and Receive from the next task \n", taskid);

                // sorry but it overlaps
                if (vec[end_index] > next_val) { 
                    is_local_sorted = false;
                    int next_len = load + ((taskid+1) < extra ? 1 : 0);
                    // printf("[Next Len] My task ID is %d, next len is %d \n", taskid, next_len);

                    // printf("My task ID is %d, expect to Receive from the next task \n", taskid);
                    MPI_Recv(&vec[end_index+1], next_len, MPI_INT, taskid + 1, 0, MPI_COMM_WORLD, status); // receive the batch from next process
                    // printf("My task ID is %d, Received from the next task, going to sort the merged array \n", taskid);
                    localOddEvenSort(vec, start_index, end_index + next_len);
                    // printf("My task ID is %d, Sorted the merged array, going to Send the next apart to next proc \n", taskid);
                    // printf("[Send Sorted] My task ID is %d, Sorted the merged array, the first element to send is %d\n", taskid, vec[end_index+1]);
                    MPI_Send(&vec[end_index+1], next_len, MPI_INT, taskid + 1, 0, MPI_COMM_WORLD); // Send it back the the next process
                }
            }

            // odd-indexed tasks
            else if (taskid % 2 == 1)  {
               if ((numtasks % 2 == 0) || (taskid != numtasks - 1)) {
                    int prev_val;
                    // printf("My task ID is %d, expect to Send and Receive from the previous task \n", taskid);
                    MPI_Sendrecv(&vec[start_index], 1, MPI_INT, taskid - 1, 0,
                                &prev_val, 1, MPI_INT, taskid - 1, 0,
                                MPI_COMM_WORLD, status);
                    // printf("My task ID is %d, Successfully Send and Receive from the previous task \n", taskid);
                    
                    // sorry but it overlaps
                    if (prev_val > vec[start_index]) {
                        is_local_sorted = false;
                        // send, receive, and replace
                        // printf("My task ID is %d, expect to Send and Receive and Replace from the previous task \n", taskid);
                        MPI_Sendrecv_replace(&vec[start_index], local_n, MPI_INT, taskid-1, 0, taskid-1, 0, MPI_COMM_WORLD, status);
                        // printf("My task ID is %d, Sucessfully Send and Receive and Replace from the previous task \n", taskid);

                    }
                }
            } 
        }

        // Odd phase
        else {
            if (numtasks == 2) {
                is_local_sorted = false;
                continue;
            }
            // odd-indexed tasks
            if (taskid % 2 == 1 && taskid != numtasks - 1) { 
                int next_val;
                // printf("My task ID is %d, expect to Send and Receive from the next task \n", taskid);

                MPI_Sendrecv(&vec[end_index], 1, MPI_INT, taskid + 1, 0,
                            &next_val, 1, MPI_INT, taskid + 1, 0,
                            MPI_COMM_WORLD, status);

                // printf("My task ID is %d, Successfully Send and Receive from the next task \n", taskid);

                // sorry but it overlaps
                if (vec[end_index] > next_val) { 
                    is_local_sorted = false;
                    int next_len = load + ((taskid+1) < extra ? 1 : 0);
                    // printf("My task ID is %d, expect to Receive from the next task \n", taskid);
                    MPI_Recv(&vec[end_index+1], next_len, MPI_INT, taskid + 1, 0, MPI_COMM_WORLD, status); // receive the batch from next process
                    // printf("My task ID is %d, Received from the next task, going to sort the merged array \n", taskid);
                    localOddEvenSort(vec, start_index, end_index + next_len);
                    // printf("My task ID is %d, Sorted the merged array, going to Send the next apart to next proc \n", taskid);
                    MPI_Send(&vec[end_index+1], next_len, MPI_INT, taskid + 1, 0, MPI_COMM_WORLD); // Send it back the the next process
                }
            }
            // even-indexed tasks
            else if (taskid % 2 == 0 && taskid != 0) {

                int prev_val;
                // printf("My task ID is %d, expect to Send and Receive from the previous task \n", taskid);

                MPI_Sendrecv(&vec[start_index], 1, MPI_INT, taskid - 1, 0,
                            &prev_val, 1, MPI_INT, taskid - 1, 0,
                            MPI_COMM_WORLD, status);
                // printf("My task ID is %d, Successfully Send and Receive from the previous task \n", taskid);

                // sorry but it overlaps
                if (prev_val > vec[start_index]) {
                    is_local_sorted = false;
                    // send, wait, receive, and replace
                    // printf("My task ID is %d, expect to Send and Receive and Replace from the previous task \n", taskid);

                    MPI_Sendrecv_replace(&vec[start_index], local_n, MPI_INT, taskid-1, 0, taskid-1, 0, MPI_COMM_WORLD, status);
                    // printf("My task ID is %d, Sucessfully Send and Receive and Replace from the previous task \n", taskid);

                }
            } 
        }

        // printf("My task ID is %d, checking condition \n", taskid);
        // Check if local array is sorted
        int local_sorted = is_local_sorted ? 1 : 0;
        int global_sorted;
        MPI_Reduce(&local_sorted, &global_sorted, 1, MPI_INT, MPI_LAND, MASTER, MPI_COMM_WORLD);
        
        MPI_Bcast(&global_sorted, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
        // printf("My task ID is %d, checked condition \n", taskid);
        if (global_sorted) {
            break;
        }
    }

    // Gather sorted subarrays at master
    std::vector<int> recv_counts(numtasks);
    std::vector<int> displs(numtasks);

    for (int i = 0; i < numtasks; ++i) {
        recv_counts[i] = (n / numtasks) + (i < n % numtasks);
        displs[i] = (i > 0) ? (displs[i - 1] + recv_counts[i - 1]) : 0;
    }

    std::vector<int> sorted_vec;
    if (taskid == MASTER) {
        sorted_vec.resize(n);
    }
    MPI_Gatherv(vec.data() + start_index, local_n, MPI_INT,
                sorted_vec.data(), recv_counts.data(), displs.data(),
                MPI_INT, MASTER, MPI_COMM_WORLD);

    if (taskid == MASTER) {
        vec = std::move(sorted_vec);
    }
}


int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 2) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable vector_size\n"
            );
    }
    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // What's my rank?
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    // Which node am I running on?
    int len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;

    const int size = atoi(argv[1]);

    const int seed = 4005;

    std::vector<int> vec = createRandomVec(size, seed);
    // std::vector<int> vec = {7, 6, 5, 4, 3, 2, 1, 0};
    std::vector<int> vec_clone = vec;

    auto start_time = std::chrono::high_resolution_clock::now();

    oddEvenSort(vec, numtasks, taskid, &status);

    if (taskid == MASTER) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Odd-Even Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        checkSortResult(vec_clone, vec);

        // std::cout << "The Sorted vector elements are : "<< std::endl;
        // for(int i=0; i < vec.size(); i++)
        //     std::cout << vec.at(i) << ' ';

        // std::cout << "\nThe Original vector elements are : "<< std::endl;
        // for(int i=0; i < vec_clone.size(); i++)
        //     std::cout << vec_clone.at(i) << ' ';

    }

    MPI_Finalize();
    return 0;
}