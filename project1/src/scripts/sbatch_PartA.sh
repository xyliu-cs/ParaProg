#!/bin/bash
#SBATCH -o ./Project1-PartA-Results.txt
#SBATCH -p Debug
#SBATCH -D /nfsmnt/120040051/CSC4005-2023Fall/project1
#SBATCH -J Project1-PartA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

# Get the current directory
CURRENT_DIR=$(cd $(dirname $0); pwd)
echo "Current directory: ${CURRENT_DIR}"

# Sequential PartA
echo "Sequential PartA (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 ./build/src/cpu/sequential_PartA ./images/20K-RGB.jpg ./images/20K-Gray.jpg
echo ""

# SIMD PartA
echo "SIMD(AVX2) PartA (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 ./build/src/cpu/simd_PartA ./images/20K-RGB.jpg ./images/20K-Gray.jpg
echo ""

# MPI PartA
echo "MPI PartA (Optimized with -O2)"
for num_processes in 1 2 4 8 16 32
do
  echo "Number of processes: $num_processes"
  srun -n $num_processes --cpus-per-task 1 --mpi=pmi2 ./build/src/cpu/mpi_PartA ./images/20K-RGB.jpg ./images/20K-Gray.jpg
  echo ""
done

# Pthread PartA
echo "Pthread PartA (Optimized with -O2)"
for num_cores in 1 2 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores ./build/src/cpu/pthread_PartA ./images/20K-RGB.jpg ./images/20K-Gray.jpg ${num_cores}
  echo ""
done

# OpenMP PartA
echo "OpenMP PartA (Optimized with -O2)"
for num_cores in 1 2 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores ./build/src/cpu/openmp_PartA ./images/20K-RGB.jpg ./images/20K-Gray.jpg
  echo ""
done

# CUDA PartA
echo "CUDA PartA"
srun -n 1 --gpus 1 ./build/src/gpu/cuda_PartA ./images/20K-RGB.jpg ./images/20K-Gray.jpg
echo ""

# OpenACC PartA
echo "OpenACC PartA"
srun -n 1 --gpus 1 ./build/src/gpu/openacc_PartA ./images/20K-RGB.jpg ./images/20K-Gray.jpg
echo ""
