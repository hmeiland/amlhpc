# aml-slurm

Package to provide a -just enough- Slurm experience on Azure Machine Learning. Use the infamous sbatch/sinfo/squeue to submit
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

Submit a job, either as a command through the wrap option or a script. sbatch uses several options, which are explained in sbatch --help.
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
