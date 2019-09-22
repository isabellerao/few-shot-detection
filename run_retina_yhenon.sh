#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
#SBATCH -p gpu
#SBATCH --gres gpu:2
# set max wallclock time
#SBATCH --time=0:05:00
# set name of job
#SBATCH --job-name=retinanet_train_retina_yhenon
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=isarao@stanford.edu

# srun nvidia-smi

ml cuda/9.2.88
ml load py-pytorch/1.0.0_py36

source ../virtualenvs/retina_yhenon/bin/activate
pip install --user torch torchvision
python3 train.py --dataset coco --coco_path /scratch/users/isarao/coco --depth 50
