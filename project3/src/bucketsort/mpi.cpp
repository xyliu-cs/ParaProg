//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Bucket Sort with MPI
//

#include <iostream>
#include <vector>
#include <mpi.h>
#include "../utils.hpp"

#define MASTER 0

void insertionSort(std::vector<int>& bucket) {
    for (int i = 1; i < bucket.size(); ++i) {
        int key = bucket[i];
        int j = i - 1;

        while (j >= 0 && bucket[j] > key) {
            bucket[j + 1] = bucket[j];
            j--;
        }

        bucket[j + 1] = key;
    }
}


void bucketSort(std::vector<int>& vec, int num_buckets, int numtasks, int taskid, MPI_Status* status) {
    int max_val = *std::max_element(vec.begin(), vec.end());
    int min_val = *std::min_element(vec.begin(), vec.end());

    int range = max_val - min_val + 1;
    int small_bucket_range = range / num_buckets;
    int large_bucket_range = small_bucket_range + 1;
    int large_bucket_num = range - small_bucket_range * num_buckets;
    int boundary = min_val + large_bucket_num * large_bucket_range; // large first, then small

    // allocate buckets
    int bucket_per_task = num_buckets / numtasks;
    int extra_buckets = num_buckets % numtasks;
    int start_bucket, end_bucket;

    // allocate buckets
    if (taskid < extra_buckets) {
        start_bucket = taskid * (bucket_per_task + 1);
        end_bucket = start_bucket + bucket_per_task + 1;
    }
    else {
        start_bucket = taskid * bucket_per_task + extra_buckets;
        end_bucket = start_bucket + bucket_per_task;
    }


    std::vector<std::vector<int>> buckets(num_buckets);
    std::vector<int> local_buckets_flat;
    local_buckets_flat.reserve((bucket_per_task+1) * large_bucket_range);


    // Pre-allocate space to avoid re-allocation
    for (int i = start_bucket; i < end_bucket; i++) {
        buckets[i].reserve(large_bucket_range);
    }


    // Place each element in the appropriate bucket
    for (int num : vec) {
        int bucket_index;
        if (num < boundary) {
            bucket_index = (num - min_val) / large_bucket_range;
        } else {
            bucket_index = large_bucket_num + (num - boundary) / small_bucket_range;
        }
        if (bucket_index >= num_buckets) {
            // Handle elements at the upper bound
            bucket_index = num_buckets - 1;
        }

        // only care about one's own buckets
        if (bucket_index >= start_bucket && bucket_index < end_bucket)
            buckets[bucket_index].push_back(num);
    }


    // sort and flat buckets for each proc
    for (int i = start_bucket; i < end_bucket; i++) {
        insertionSort(buckets[i]);
        local_buckets_flat.insert(local_buckets_flat.end(), buckets[i].begin(), buckets[i].end());
    }


    // Gather bucket sizes at the master
    int local_size = local_buckets_flat.size();
    std::vector<int> all_sizes(numtasks);
    std::vector<int> displs(numtasks, 0);   
    std::vector<int> sorted_vec;

    MPI_Gather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Calculate displacements
    if (taskid == MASTER) {
        for (int i = 1; i < numtasks; ++i) {
            displs[i] = displs[i - 1] + all_sizes[i - 1];
        }
        sorted_vec.resize(vec.size());
    }

    MPI_Gatherv(local_buckets_flat.data(), local_buckets_flat.size(), MPI_INT, 
                sorted_vec.data(), all_sizes.data(), displs.data(), MPI_INT, MASTER, MPI_COMM_WORLD);

    if (taskid == MASTER) {
        vec = std::move(sorted_vec);
    }

}



int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable vector_size bucket_num\n"
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

    const int bucket_num = atoi(argv[2]);

    const int seed = 4005;

    std::vector<int> vec = createRandomVec(size, seed);
    std::vector<int> vec_clone = vec;

    auto start_time = std::chrono::high_resolution_clock::now();

    bucketSort(vec, bucket_num, numtasks, taskid, &status);

    if (taskid == MASTER) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Bucket Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}