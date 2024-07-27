#!/bin/bash
#SBATCH --account=def-gerope
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=1-00:00
#SBATCH --output=prod_output_ros/%N-%j.out

module --force purge
module load StdEnv/2023
module load python/3.10
module load cuda/12.2
source /home/jcchen/projects/def-gerope/jcchen/gemini/ENV/bin/activate

python label-data.py 25