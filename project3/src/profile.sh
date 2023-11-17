#!/bin/bash
#SBATCH -o ./Project3-Profile-Log.txt
#SBATCH -p Project
#SBATCH -J Project3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1


# mkdir ./profiling

# Array of core counts
CORE_COUNTS=(1 2 4 8 16 32)

# Output file
OUTPUT_FILE="./profiling/profiling_results.txt"


echo "[quicksort_mpi]" >> $OUTPUT_FILE
for cores in "${CORE_COUNTS[@]}"; do
    echo "Running with ${cores} cores" >> $OUTPUT_FILE
    srun -n ${cores} --cpus-per-task=1 --mpi=pmi2 perf stat -e cpu-cycles,cache-misses,page-faults ./build/src/quicksort/quicksort_mpi 100000000 2>> $OUTPUT_FILE
    echo "--------------------------------------" >> $OUTPUT_FILE
done


echo "[bucketsort_mpi]" >> $OUTPUT_FILE
for cores in "${CORE_COUNTS[@]}"; do
    echo "Running with ${cores} cores" >> $OUTPUT_FILE
    srun -n ${cores} --cpus-per-task=1 --mpi=pmi2 perf stat -e cpu-cycles,cache-misses,page-faults ./build/src/bucketsort/bucketsort_mpi 100000000 2500000 2>> $OUTPUT_FILE
    echo "--------------------------------------" >> $OUTPUT_FILE
done

echo "[odd-even-sort_mpi]" >> $OUTPUT_FILE
for cores in "${CORE_COUNTS[@]}"; do
    echo "Running with ${cores} cores" >> $OUTPUT_FILE
    srun -n ${cores} --cpus-per-task=1 --mpi=pmi2 perf stat -e cpu-cycles,cache-misses,page-faults ./build/src/odd-even-sort/odd-even-sort_mpi 200000 2>> $OUTPUT_FILE
    echo "--------------------------------------" >> $OUTPUT_FILE
done
