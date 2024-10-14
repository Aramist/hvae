#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -c 16
#SBATCH -G 1
#SBATCH -C a100-80gb
#SBATCH --mem=64GB
#SBATCH --time=2-0
#SBATCH -o /mnt/home/atanelus/hvae_%j.log
pwd; hostname; date;

module purge;
source /mnt/home/atanelus/.bashrc
source /mnt/home/atanelus/venvs/new/bin/activate

sp=$(basename $1)
python -m hvae --data /mnt/home/atanelus/ceph/hvae_dataset_1ms.hdf5 --save-path /mnt/home/atanelus/ceph/hvae_models_1ms/$sp --config $1