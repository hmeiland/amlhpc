# aml-slurm

Package to provide a -just enough- Slurm experience on Azure Machine Learning. Use the infamous sbatch/sinfo/squeue to submit
jobs and get insight into the state of the HPC system through a familiar way. Allow applications to interact with AML without 
the need to re-program another integration.

# sinfo

Show the available partitions. sinfo does not take any options.

# squeue

Show the queue with historical jobs. squeue does not take any options.

# sbatch

Submit a job, either as a command through the wrap option or a script. sbatch uses several options, which are explained in sbatch --help.
