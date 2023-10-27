#!/bin/bash
#SBATCH -o ./Project2-Results.txt
#SBATCH -p Project
#SBATCH -J Project2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

CURRENT_DIR=$(pwd)/src

# Naive
echo "Naive Matrix Multiplication 1024 (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/naive1.txt
echo ""

# echo "Naive Matrix Multiplication abnormal (Optimized with -O2)"
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/naive ${CURRENT_DIR}/../matrices/matrix_ab1.txt ${CURRENT_DIR}/../matrices/matrix_ab2.txt ${CURRENT_DIR}/../results/naive_ab.txt
# echo ""




# Memory Locality
echo "Memory Locality Matrix Multiplication 1024 (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/locality1.txt
echo ""

# echo "Memory Locality Matrix Multiplication 2048 (Optimized with -O2)"
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/locality2.txt
# echo ""

# echo "Memory Locality Matrix Multiplication abnormal (Optimized with -O2)"
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/locality ${CURRENT_DIR}/../matrices/matrix_ab1.txt ${CURRENT_DIR}/../matrices/matrix_ab2.txt ${CURRENT_DIR}/../results/locality_ab.txt
# echo ""




# # SIMD + Reordering
echo "SIMD + Memory Locality Matrix Multiplication 1024 (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/simd ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/simd1.txt
echo ""

# echo "SIMD + Memory Locality Matrix Multiplication 2048 (Optimized with -O2)"
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/simd ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/simd2.txt
# echo ""

# echo "SIMD + Memory Locality Matrix Multiplication abnormal (Optimized with -O2)"
# srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../build/src/simd ${CURRENT_DIR}/../matrices/matrix_ab1.txt ${CURRENT_DIR}/../matrices/matrix_ab2.txt ${CURRENT_DIR}/../results/simd_ab.txt
# echo ""




# OpenMP + SIMD + Reordering
echo "OpenMP + SIMD + Memory Locality Matrix Multiplication 1024 (Optimized with -O2)"
for num_cores in 1 2 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/openmp1.txt
  echo ""
done

# echo "OpenMP + SIMD + Memory Locality Matrix Multiplication 2048 (Optimized with -O2)"
# for num_cores in 1 2 4 8 16 32
# do
#   echo "Number of cores: $num_cores"
#   srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/openmp2.txt
#   echo ""
# done

# echo "OpenMP + SIMD + Memory Locality Matrix Multiplication abnormal (Optimized with -O2)"
# for num_cores in 1 2 4 8 16 32
# do
#   echo "Number of cores: $num_cores"
#   srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../build/src/openmp $num_cores ${CURRENT_DIR}/../matrices/matrix_ab1.txt ${CURRENT_DIR}/../matrices/matrix_ab2.txt ${CURRENT_DIR}/../results/openmp_ab.txt
#   echo ""
# done




# # MPI + OpenMP + SIMD + Reordering
echo "MPI + OpenMP + SIMD + Memory Locality Matrix Multiplication 1024 (Optimized with -O2)"
echo "Number of Processes: 1, Number of Threads: 32"
srun -n 1 --cpus-per-task 32 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 32 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/mpi1.txt
echo ""

echo "Number of Processes: 2, Number of Threads: 16"
srun -n 2 --cpus-per-task 16 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 16 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/mpi1.txt
echo ""

echo "Number of Processes: 4, Number of Threads: 8"
srun -n 4 --cpus-per-task 8 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 8 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/mpi1.txt
echo ""

echo "Number of Processes: 8, Number of Threads: 4"
srun -n 8 --cpus-per-task 4 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 4 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/mpi1.txt
echo ""

echo "Number of Processes: 16, Number of Threads: 2"
srun -n 16 --cpus-per-task 2 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 2 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/mpi1.txt
echo ""

echo "Number of Processes: 32, Number of Threads: 1"
srun -n 32 --cpus-per-task 1 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 1 ${CURRENT_DIR}/../matrices/matrix5.txt ${CURRENT_DIR}/../matrices/matrix6.txt ${CURRENT_DIR}/../results/mpi1.txt
echo ""


# echo "MPI + OpenMP + SIMD + Memory Locality Matrix Multiplication 2048 (Optimized with -O2)"
# echo "Number of Processes: 1, Number of Threads: 32"
# srun -n 1 --cpus-per-task 32 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 32 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/mpi2.txt
# echo ""

# echo "Number of Processes: 2, Number of Threads: 16"
# srun -n 2 --cpus-per-task 16 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 16 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/mpi2.txt
# echo ""

# echo "Number of Processes: 4, Number of Threads: 8"
# srun -n 4 --cpus-per-task 8 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 8 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/mpi2.txt
# echo ""

# echo "Number of Processes: 8, Number of Threads: 4"
# srun -n 8 --cpus-per-task 4 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 4 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/mpi2.txt
# echo ""

# echo "Number of Processes: 16, Number of Threads: 2"
# srun -n 16 --cpus-per-task 2 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 2 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/mpi2.txt
# echo ""

# echo "Number of Processes: 32, Number of Threads: 1"
# srun -n 32 --cpus-per-task 1 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 1 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/mpi2.txt
# echo ""


# echo "MPI + OpenMP + SIMD + Memory Locality Matrix Multiplication abnormal (Optimized with -O2)"
# echo "Number of Processes: 16, Number of Threads: 2"
# srun -n 16 --cpus-per-task 2 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 2 ${CURRENT_DIR}/../matrices/matrix_ab1.txt ${CURRENT_DIR}/../matrices/matrix_ab2.txt ${CURRENT_DIR}/../results/mpi_ab.txt
# echo ""


# MPI 64 threads tests
# echo "MPI + OpenMP + SIMD + Memory Locality Matrix Multiplication 2048 (Optimized with -O2)"
# echo "Number of Processes: 1, Number of Threads: 64"
# srun -n 1 --cpus-per-task 64 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 64 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/mpi2.txt
# echo ""

# echo "Number of Processes: 2, Number of Threads: 32"
# srun -n 2 --cpus-per-task 32 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 32 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/mpi2.txt
# echo ""

# echo "Number of Processes: 4, Number of Threads: 16"
# srun -n 4 --cpus-per-task 16 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 16 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/mpi2.txt
# echo ""

# echo "Number of Processes: 8, Number of Threads: 8"
# srun -n 8 --cpus-per-task 8 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 8 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/mpi2.txt
# echo ""

# echo "Number of Processes: 16, Number of Threads: 4"
# srun -n 16 --cpus-per-task 4 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 4 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/mpi2.txt
# echo ""

# echo "Number of Processes: 32, Number of Threads: 2"
# srun -n 32 --cpus-per-task 2 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 2 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/mpi2.txt
# echo ""

# echo "Number of Processes: 64, Number of Threads: 1"
# srun -n 64 --cpus-per-task 1 --mpi=pmi2 ${CURRENT_DIR}/../build/src/mpi 1 ${CURRENT_DIR}/../matrices/matrix7.txt ${CURRENT_DIR}/../matrices/matrix8.txt ${CURRENT_DIR}/../results/mpi2.txt
# echo ""