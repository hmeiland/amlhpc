# amlhpc

📖 Full documentation: https://amlhpc.readthedocs.io/en/latest/index.html

Package to provide a -just enough- Slurm or PBS experience on Azure Machine Learning. Use the infamous sbatch/qsub/sinfo to submit
jobs and get insight into the state of the HPC system through a familiar way. Allow applications to interact with AML without 
the need to re-program another integration.

See [architecture.md](architecture.md) for how amlhpc integrates with Azure Machine Learning.

For the commands to function, the following environment variables have to be set:
```
SUBSCRIPTION=<guid of you Azure subscription e.g. 12345678-1234-1234-1234-1234567890ab>
CI_RESOURCE_GROUP=<name of the resource group where your Azure Machine Learning Workspace is created>
CI_WORKSPACE=<name of your Azure MAchine Learning Workspace>
```

In the Azure Machine Learning environment, the CI_RESOURCE_GROUP and CI_WORKGROUP are normally set, so you only need to export SUBSCRIPTION.

Every command is available both as a standalone executable (`sbatch`, `srun`, `sinfo`, `squeue`, `deploy`, ...)
and as a subcommand of the unified `amlhpc` entry point, so `amlhpc sbatch ...` is equivalent to `sbatch ...`.
```
(azureml_py38) azureuser@login-vm:~$ amlhpc --help
usage: amlhpc <command> [<args>]

amlhpc: a -just enough- Slurm/PBS experience on Azure Machine Learning

commands:
  sbatch
  srun
  sinfo
  squeue
  scancel
  sacct
  sstat
  sattach
  qsub
  qstat
  qdel
  bjobs
  bkill
  bsub
  container
  deploy
  dask-scheduler-up
  dask-up
  dask-down

run 'amlhpc <command> --help' for command-specific options
```

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

# squeue / qstat / bjobs

Show the queue with historical jobs, including the job STATE. `squeue` is the Slurm front-end;
`qstat` (PBS) and `bjobs` (LSF) are the same listing for users with different scheduler muscle
memory. All three are backed by the same Azure Machine Learning job list and take no options
(`squeue` no longer pauses for a keypress between pages). `squeue` additionally prints a TIME
column; `qstat`/`bjobs` produce otherwise identical output.
```
(azureml_py38) azureuser@login-vm:~/cloudfiles/code/Users/username$ squeue
JOBID                           NAME            PARTITION       STATE   TIME
crimson_root_52y4l9yfjd         sbatch  	f16s	Completed
polite_lock_v8wyc9gnx9          runscript.sh    f16s	Running
```
```
(azureml_py38) azureuser@login-vm:~/cloudfiles/code/Users/username$ qstat
JOBID                           NAME            PARTITION       STATE
jolly_card_p6yh0phzxm           jolly_card_p6yh login-5n2kkmvhk Completed
cool_pig_mwhdcjs72n             localtest-f4s   f4s             Failed
```

# qdel / bkill / scancel

Cancel one or more running jobs by JOBID. `qdel` (PBS), `bkill` (LSF) and `scancel` (Slurm) are
equivalent; all accept one or more JOBIDs (as shown by `qstat`/`bjobs`/`squeue`).
```
(azureml_py38) azureuser@login-vm:~/cloudfiles/code/Users/username$ qdel jolly_card_p6yh0phzxm
qdel: cancellation requested for job 'jolly_card_p6yh0phzxm'
```
```
(azureml_py38) azureuser@login-vm:~/cloudfiles/code/Users/username$ bkill jobid1 jobid2
```
```
(azureml_py38) azureuser@login-vm:~/cloudfiles/code/Users/username$ scancel wheat_sand_gr2xcdpl2w
scancel: cancellation requested for job 'wheat_sand_gr2xcdpl2w'
```

# sacct

Show the status of one or more specific jobs by JOBID. Where `squeue`/`qstat` list every job,
`sacct` targets the JOBIDs you name and adds the start/end timestamps from the job's lifecycle,
so it is the quick "is this job done, and how long did it take?" lookup.
```
(azureml_py38) azureuser@login-vm:~/cloudfiles/code/Users/username$ sacct polite_kitten_x3g0kwt9w2
JOBID                           NAME            PARTITION       STATE           START               END
polite_kitten_x3g0kwt9w2        polite_kitten_x f4s             Completed       2026-07-18T11:49:45 2026-07-18T12:36:08
```
Multiple JOBIDs may be given; a JOBID that does not exist is reported and `sacct` exits non-zero.

# sstat

Show the node CPU/memory utilization for a job over the window it ran. Utilization is an Azure
Monitor platform metric on the *workspace* resource, so it is **whole-node and not job-scoped**:
on a shared cluster running several jobs the figures cover the entire node, not just your job.
By default `sstat` prints the latest sample and the peak; `--history` prints the full per-minute
series. Requires the `azure-monitor-query` package (installed as a dependency).
```
(azureml_py38) azureuser@login-vm:~/cloudfiles/code/Users/username$ sstat polite_kitten_x3g0kwt9w2
job polite_kitten_x3g0kwt9w2 node utilization (whole-node, over job window; not job-scoped)
  CpuUtilizationPercentage: latest avg=84.0%  peak max=84.0% @ 2026-07-18 12:36:00  (49 samples)
  CpuMemoryUtilizationPercentage: no data
```
```
(azureml_py38) azureuser@login-vm:~/cloudfiles/code/Users/username$ sstat --history polite_kitten_x3g0kwt9w2
  CpuUtilizationPercentage (avg/max per minute):
    2026-07-18 11:52:00  avg=2.0%  max=2.0%
    2026-07-18 11:53:00  avg=3.0%  max=3.0%
    ...
```

# sattach

Show or follow the log output of a job by JOBID. By default `sattach` prints the job's `std_log`
one-shot (handy after a job has finished); with `-f`/`--follow` it streams the log until the job
completes, like `tail -f`.
```
(azureml_py38) azureuser@login-vm:~/cloudfiles/code/Users/username$ sattach polite_kitten_x3g0kwt9w2
313/313 - 2s - loss: 0.8838 - accuracy: 0.7076 - 2s/epoch - 6ms/step
0.7075999975204468
```
```
(azureml_py38) azureuser@login-vm:~/cloudfiles/code/Users/username$ sattach -f amusing_frame_30vvwy3jc3
```

# qsub

Submit a job the PBS way. `qsub` is a thin, PBS-style front-end for `sbatch`: `-q` selects the queue,
`-l nodes=N` (or `select=N`, or `nodes=N:ppn=P`) sets the node count, and `-N` is an informational
job name. A single existing file is treated as a script; anything else is run as a command line.
```
(azureml_py38) azureuser@login-vm:~/cloudfiles/code/Users/username$ qsub -q f16s hostname
gifted_engine_yq801rygm2
```
```
(azureml_py38) azureuser@login-vm:~/cloudfiles/code/Users/username$ qsub -q hbv2 -l nodes=4 ./runscript.sh
```
PBS-style options map onto sbatch: `-q/--queue` -> partition, `-l nodes=/select=` -> nodes,
`-N/--name` is informational (AML assigns the JOBID). `--container`/`-e` pass through.

# bsub

Submit a job the LSF way. `bsub` is a thin, LSF-style front-end for `sbatch`: the job command
follows the options (rather than using `--wrap`), and `-q` selects the queue. A single existing
file is treated as a script; anything else is run as a command line.
```
(azureml_py38) azureuser@login-vm:~/cloudfiles/code/Users/username$ bsub -q f16s hostname
gifted_engine_yq801rygm2
```
```
(azureml_py38) azureuser@login-vm:~/cloudfiles/code/Users/username$ bsub -q hbv2 -n 4 ./runscript.sh
```
LSF-style options map onto sbatch: `-q/--queue` -> partition, `-n/--num-slots` -> nodes,
`-J/--job-name` is informational (AML assigns the JOBID). `--container`/`-e` pass through.

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
usage: sbatch [-h] [-a ARRAY] [--container CONTAINER] [--datamover DATAMOVER]
              [-e ENVIRONMENT] [-N NODES] [-p PARTITION] [-v] [--no-prolog]
              [-w WRAP]
              [script]

sbatch: submit jobs to Azure Machine Learning

positional arguments:
  script                runscript to be executed

options:
  -h, --help            show this help message and exit
  -a ARRAY, --array ARRAY
                        index for array jobs
  --container CONTAINER
                        container environment for the job to run in
  --datamover DATAMOVER
                        use "simple" for moving the (recursive) data along
                        with the runscript
  -e ENVIRONMENT, --environment ENVIRONMENT
                        Azure Machine Learning environment, should be enclosed
                        in quotes, may use @latest
  -N NODES, --nodes NODES
                        amount of nodes to use for the job
  -p PARTITION, --partition PARTITION
                        set compute partition where the job should be run. Use <sinfo> to view available partitions
  -v, --verbose         provide output on found settings and job properties
  --no-prolog           skip the site-wide prolog/epilog configured in the
                        workspace storage stack
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
usage: srun [-h] [--container CONTAINER] [-e ENVIRONMENT] [-p PARTITION] [-v]
            [--no-prolog] [-w WRAP]
            [script]

srun: run jobs on the login ComputeInstance

positional arguments:
  script                runscript to be executed

options:
  -h, --help            show this help message and exit
  --container CONTAINER
                        container image for the job to run in
  -e ENVIRONMENT, --environment ENVIRONMENT
                        Azure Machine Learning environment, may use @latest
  -p PARTITION, --partition PARTITION
                        ComputeInstance to run on. Defaults to auto-discovered login CI. Use <sinfo> to view partitions
  -v, --verbose         provide output on found settings and job properties
  --no-prolog           skip the site-wide prolog/epilog configured in the
                        workspace storage stack
  -w WRAP, --wrap WRAP  command line to be executed, should be enclosed with quotes
```

Note: like every AML command job, the command runs inside the environment's container (sharing the CI's
network namespace, so it sees the CI private IP), not the bare CI host OS.

# site-wide prolog/epilog

An admin can install a site-wide prolog (and optional epilog) once into the workspace storage stack;
every `sbatch`/`srun` job then runs it automatically as `prolog → ( user command ) → epilog`, so users
can submit bare application commands without repeating boilerplate (mounting a shared software stack,
loading a module, sourcing an environment). The user command runs in a subshell, so a failing or
`exit`-ing command still runs the epilog and its exit code is preserved. Pass `--no-prolog` to opt a
single job out.

The hooks are managed with `deploy config` (see [deploy/README.md](deploy/README.md)) and live in the
workspace's default datastore at `workspaceblobstore/amlhpc/{prolog,epilog}.sh` (override the location
with `AMLHPC_CONFIG_DATASTORE` / `AMLHPC_CONFIG_PREFIX`). A worked EESSI example is in
[examples/EESSI/readme.md](examples/EESSI/readme.md).
```
(azureml_py38) azureuser@login-vm:~/cloudfiles/code/Users/username$ deploy config set-prolog prolog.sh
uploaded site prolog to azureml://datastores/workspaceblobstore/paths/amlhpc/prolog.sh
(azureml_py38) azureuser@login-vm:~/cloudfiles/code/Users/username$ sbatch -p f4s --wrap="simpleFoam"
gifted_engine_yq801rygm2
```

If you encounter a scenario or option that is not supported yet or behaves unexpected, please create an issue and explain the option and the scenario.
