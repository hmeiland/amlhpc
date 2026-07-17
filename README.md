# amlhpc

Package to provide a -just enough- Slurm or PBS experience on Azure Machine Learning. Use the infamous sbatch/qsub/sinfo to submit
jobs and get insight into the state of the HPC system through a familiar way. Allow applications to interact with AML without 
the need to re-program another integration.

For the commands to function, the following environment variables have to be set:
```
SUBSCRIPTION=<guid of you Azure subscription e.g. 12345678-1234-1234-1234-1234567890ab>
CI_RESOURCE_GROUP=<name of the resource group where your Azure Machine Learning Workspace is created>
CI_WORKSPACE=<name of your Azure MAchine Learning Workspace>
```

In the Azure Machine Learning environment, the CI_RESOURCE_GROUP and CI_WORKGROUP are normally set, so you only need to export SUBSCRIPTION.

# sinfo

Show the available partitions. sinfo does not take any options.
```
(azureml_py38) azureuser@login-vm:~/cloudfiles/code/Users/username$ sinfo
PARTITION       AVAIL   VM_SIZE                 NODES   STATE
f16s            UP      STANDARD_F16S_V2        37
hc44            UP      STANDARD_HC44RS         3
hbv2            UP      STANDARD_HB120RS_V2     4
login-vm        UP      STANDARD_DS12_V2        None
```

# squeue

Show the queue with historical jobs. squeue does not take any options.
```
(azureml_py38) azureuser@login-vm:~/cloudfiles/code/Users/username$ squeue
JOBID                           NAME            PARTITION       STATE   TIME
crimson_root_52y4l9yfjd         sbatch  	f16s
polite_lock_v8wyc9gnx9          runscript.sh    f16s
```

# sbatch

Submit a job, either as a command through the `--wrap` option or a (shell) script. sbatch uses several options, which are explained in sbatch --help.
Quite a bit of sbatch options are supported such as running multi-node MPI jobs with the option to set the amount of nodes to be used.
Also array jobs are supported with the default `--array` option.

Some additional options are introduced to support e.g. the data-handling methods available in AML. These are explaned in [data.md](data.md). 
```
(azureml_py38) azureuser@login-vm:~/cloudfiles/code/Users/username$ sbatch -p f16s --wrap="hostname"
gifted_engine_yq801rygm2
```
```
(azureml_py38) azureuser@login-vm:~/cloudfiles/code/Users/username$ sbatch --help
usage: sbatch [-h] [-a ARRAY] -p PARTITION [-N NODES] [-w WRAP] [script]

sbatch: submit jobs to Azure Machine Learning

positional arguments:
  script                script to be executed

optional arguments:
  -h, --help            show this help message and exit
  -a ARRAY, --array ARRAY
                        index for array jobs
  -p PARTITION, --partition PARTITION
                        set compute partition where the job should be run. Use <sinfo> to view available partitions
  -N NODES, --nodes NODES
                        amount of nodes to use for the job
  -w WRAP, --wrap WRAP  command line to be executed, should be enclosed with quotes
```

# srun

Run a job on the login ComputeInstance instead of an AmlCompute cluster. `srun` mirrors `sbatch`, but the
compute target is a ComputeInstance. With no `-p` the login CI is auto-discovered (the single computeinstance
in the workspace). Use this to run control-plane tasks on the login node itself without SSH.
```
(azureml_py38) azureuser@login-vm:~/cloudfiles/code/Users/username$ srun --wrap="hostname"
stoic_beach_c4jpkltdrl
```
```
(azureml_py38) azureuser@login-vm:~/cloudfiles/code/Users/username$ srun --help
usage: srun [-h] [--container CONTAINER] [-e ENVIRONMENT] [-p PARTITION] [-v] [-w WRAP] [script]

srun: run jobs on the login ComputeInstance

positional arguments:
  script                runscript to be executed

optional arguments:
  -h, --help            show this help message and exit
  --container CONTAINER
                        container image for the job to run in
  -e ENVIRONMENT, --environment ENVIRONMENT
                        Azure Machine Learning environment, may use @latest
  -p PARTITION, --partition PARTITION
                        ComputeInstance to run on. Defaults to auto-discovered login CI. Use <sinfo> to view partitions
  -w WRAP, --wrap WRAP  command line to be executed, should be enclosed with quotes
```

Note: like every AML command job, the command runs inside the environment's container (sharing the CI's
network namespace, so it sees the CI private IP), not the bare CI host OS.

If you encounter a scenario or option that is not supported yet or behaves unexpected, please create an issue and explain the option and the scenario.
