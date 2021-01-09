#!/bin/bash

### Each node has 8 GPUs, 40 physical cores spread across 
### 2 sockets with 2 hardware threads per core, and 384 GB DRAM.


#SBATCH -C gpu -J Raytune -t 3:59:00 --account m2043
#SBATCH -N2 -c 10 --ntasks-per-node=3 -G 10


echo GPU per Ray work nodes: ${SLURM_GPUS}
echo Total number of work nodes: ${SLURM_NTASKS}

suffix='6379'
ip_head=`hostname`:$suffix
export ip_head 


srun --nodes=1 --ntasks=1 ray start --head --block --port=6379 &
sleep 10

srun --nodes=1 --ntasks=1 ray start --address $ip_head --block &
sleep 5

python ./train_RayTune.py --dataPath /global/homes/b/balewski/prjn/neuronBBP-pack40kHzDisc/probe_quad/bbp153 --probeType quad -t 13500 --useDataFrac 1 --steps 10 --rayResult $SCRATCH/ray_results --numHparams 30