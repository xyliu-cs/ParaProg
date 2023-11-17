//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Quick Sort with MPI
//


#include <iostream>
#include <vector>
#include <queue>
#include <mpi.h>
#include "../utils.hpp"

#define MASTER 0

using HeapElement = std::pair<int, int>;

int partition(std::vector<int> &vec, int low, int high) {
    int pivot = vec[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (vec[j] <= pivot) {
            i++;
            std::swap(vec[i], vec[j]);
        }
    }

    std::swap(vec[i + 1], vec[high]);
    return i + 1;
}

void localQuickSort(std::vector<int>& vec, int low, int high) {
    if (low < high) {
        // Partition the array and get the pivot element index
        int pi = partition(vec, low, high);

        // Recursively sort elements before and after partition
        localQuickSort(vec, low, pi - 1);
        localQuickSort(vec, pi + 1, high);
    }
}


void quickSort(std::vector<int>& vec, int numtasks, int taskid, MPI_Status* status) {
    // Determine the size of the portion each process will sort
    int n = vec.size();
    int local_size = n / numtasks;
    int extra = n % numtasks;
    int start, end;
    int start_idx[numtasks];
    int lens[numtasks];


    for (int id = 0; id < numtasks; id++) {
        if (id < extra) {
            start = id * (local_size + 1);
            end = start + local_size;
        } else {
            start = id * local_size + extra;
            end = start + (local_size - 1);
        }
        start_idx[id] = start;
        lens[id] = end - start + 1;
    }


    // Each process sorts its portion of the array
    localQuickSort(vec, start_idx[taskid], start_idx[taskid]+lens[taskid]-1);
    
    if (numtasks == 1) {
        return;
    }
    
    // Gather all sorted subarrays at the master process
    if (taskid == MASTER) {
        std::vector<MPI_Request> recv_requests(numtasks - 1);
        for (int i = 1; i < numtasks; i++) {
            // Receive sorted subarray from each process
            MPI_Irecv(&vec[start_idx[i]], lens[i], MPI_INT, i, 0, MPI_COMM_WORLD, &recv_requests[i - 1]);
        }

        // Create a min heap
        std::priority_queue<HeapElement, std::vector<HeapElement>, std::greater<HeapElement>> minHeap;
       // Vector to track the current index of each subarray
        std::vector<int> subarrayIndices(numtasks, 0);
        //tmp vector to store sorted elements
        std::vector<int> mergedArray;
        mergedArray.reserve(vec.size());

        // Wait for all non-blocking receives to complete
        MPI_Waitall(numtasks - 1, recv_requests.data(), MPI_STATUSES_IGNORE);

        // Initialize the heap with the first element of each subarray
        for (int i = 0; i < numtasks; ++i) {
            minHeap.emplace(vec[start_idx[i]], i);           
        }

        while (!minHeap.empty()) {
            // Extract the smallest element and its subarray index
            auto topElement = minHeap.top();
            auto value = topElement.first;
            auto idx = topElement.second;
            
            minHeap.pop();

            // Add the smallest element to the merged array
            mergedArray.push_back(value);

            // Move to the next element in the subarray
            subarrayIndices[idx]++;

            // If the subarray has more elements, add the next element to the heap
            if (subarrayIndices[idx] < lens[idx]) {
                minHeap.emplace(vec[start_idx[idx] + subarrayIndices[idx]], idx);
            }
        }

        // Replace the original vector with the merged array
        vec = std::move(mergedArray);

    } else {
        // Send sorted subarray to the master process
        MPI_Send(&vec[start_idx[taskid]], lens[taskid], MPI_INT, MASTER, 0, MPI_COMM_WORLD);
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
    std::vector<int> vec_clone = vec;

    auto start_time = std::chrono::high_resolution_clock::now();

    quickSort(vec, numtasks, taskid, &status);

    if (taskid == MASTER) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Quick Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}