#!/bin/bash
#SBATCH -o ./Project1-PartB-Results.txt
#SBATCH -p Project
#SBATCH -J Project1-PartB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

# Get the current directory
CURRENT_DIR=$(pwd)/src/scripts
echo "Current directory: ${CURRENT_DIR}"


echo "OpenMP PartB (Optimized with -O2)"
for num_cores in 1 2 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../../build/src/cpu/openmp_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/openmp/20K-Smooth-$num_cores.jpg ${num_cores}
  echo ""
done