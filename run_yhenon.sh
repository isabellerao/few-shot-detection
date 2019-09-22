#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --gres gpu:1
# set max wallclock time
#SBATCH --time=4:00:00
# set name of job
#SBATCH --job-name=retinanet_train_yhenon
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=isarao@stanford.edu

# srun nvidia-smi

ml load labs poldrack anaconda/5.0.0-py36

source activate /home/users/isarao/.conda/envs/yhenon
ml cuda/9.2.88

python3 train.py --dataset coco --coco_path /scratch/users/isarao/coco --depth 50 --epochs 1000
