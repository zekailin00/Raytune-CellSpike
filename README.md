# **Ray Tune and Horovod Integration for Hyperparameter Tuning**

## **Model Setup**

This model is built for Deep_CellSpike model from https://bitbucket.org/balewski/pitchforkoracle/src/master/

To add this Hyperparameter Tuning extension, the original model is required:
```
$ cd $HOME
$ git clone https://bitbucket.org/balewski/pitchforkoracle/src/master/
$ cd master
$ cp toolbox/* .
```
After setting up the original model, copy all files in this repo to `master` :
```
$ cd $HOME
$ cp Cori/NeuronInverter/RayTuneCellSpike/* master
```
Load tensorflow/gpu-2.1.0-py37 module and install relevant packages for Python:
```
$ module load cgpu
$ module load tensorflow/gpu-2.1.0-py37

$ pip install --user 'ray[tune]'
$ pip install --user --upgrade ray==1.3.0
$ pip install --user --upgrade horovod==0.22.0
$ HOROVOD_WITH_GLOO=1
$ pip install --user 'horovod[ray]'
$ pip install --user hyperopt
```

## **Submitting Jobs**

Open a new terminal:
```
$ module load esslurm
$ cd $HOME/master
$ sbatch submit-ray-cluster.sh
```
To check the submitted jobs, `squeue -u [userName]`, e.g. `squeue -u zekailin`

To cancel a job, `scancel [jobID]`, which can be found by checking submitted jobs

## **Configuring Tuning Process**
#### `-J`
Name of Slurm submission.

#### `--time`
Wall time allocated for the Slurm job. Max 8 hours.

#### `--nodes`
Total number of GPU nodes. Usually 1 is enough. Try not to be more than 2; otherwise, submission can have a very long pending time.

#### `--ntasks-per-node`
*MUST* be 1 for Ray workers to run on each node.

#### `--gpus-per-task`
GPUs per node. Max 8 for each node. Must be less than 8 * `--nodes`.

#### `--cpus-per-task`
No need to change. Submissions will always receive 80 * `--nodes` cpus.

#### `restoreID`
To restore the hyperopt searcher state from a previous training, assign the submission ID of the training to this variable. Comment out restoreID if starting a new training.

#### `numHparams`
Number of hyperparameter sets (trials) to try. For each trial, Ray tune will generate a new set of hyperparameters either randomly or based on the hyperopt algorithm. Make sure that `--time` is large enough so that all `numHparams` trials can finish before the time limit. 40 trials for 4 hours is OK.

#### `numGPU`
Number of GPUs allocated per trial. Ray tune does parallel hyperparameter tuning, which means when `numGPU` is less than `--gpus-per-task `, Ray tune runs `--gpus-per-task `/`numGPU` trials concurrently, and Horovod uses `numGPU` to do distributed training for each trial. Try to be a factor of `--gpus-per-task `

#### `globalSamples`
Total sample size used for training each trial. Using Horovod framework, each trial has `numGPU` workers (gpus). Each worker takes a local sample size of data equal to `globalSamples`/`numGPU`.


#### `epochs`
The neural network models usually converge in 100 epochs, so smaller than 100 is OK.

## **Ray Tune & Horovod Architecture Reference**
https://horovod.readthedocs.io/en/stable/hyperparameter_search_include.html
