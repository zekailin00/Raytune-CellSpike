#!/bin/bash

### Each node has 8 GPUs, 40 physical cores spread across 
### 2 sockets with 2 hardware threads per core, and 384 GB DRAM.


#SBATCH -C gpu -J Ray-GPU -t 3:59:00 --account m2043
#SBATCH -N1 -c 10 --ntasks-per-node=3 -G 4

trainTime=20
useDataFrac=0.05
steps=10
numHparams=10

### no training configuration below this line
####################################################################

module load tensorflow/gpu-2.1.0-py37

echo Total number of GPU: ${SLURM_GPUS}

suffix='6379'
ip_head=`hostname`:$suffix
export ip_head 


srun --nodes=1 --ntasks=1 ray start --head --block --port=6379 &
sleep 10

srun --nodes=1 --ntasks=1 ray start --address $ip_head --block &
sleep 5

python ./train_RayTune.py --dataPath /global/homes/b/balewski/prjn/neuronBBP-pack40kHzDisc/probe_quad/bbp153 --probeType quad -t $trainTime --useDataFrac $useDataFrac --steps $steps --rayResult $SCRATCH/ray_results --numHparams $numHparams --nodes GPU