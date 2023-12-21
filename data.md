# Data handling

## sbatch --wrap "hostname"

For the most simple command-line job, no data is transferred from the submit host to the cluster. 
The wrapped command is forwarded and the standard-out is captured in the job output `std_log.txt` file which can be found in the `Outputs and log` tab of the job.
Any files created in the `./outputs` folder of the workdir are also captured by the job and can be found in the same tab.

## sbatch ./runscript.sh

For a job which is described in a `runscript.sh` (or any other filename), the `runscript.sh` (and only the runscript.sh)
is copied into the job and can be found in the `Code` tab of the job. 
During the job preparation it is copied into the workdir, and since the command points to this script, it will be executed. 
Any files created in the `./outputs` folder of the workdir are also captured by the job and can be found in the same tab.

## sbatch --datamover=simple ./runscript

When the datamover=simple option is added, all files in the current directory will be added to the job's `Code` tab and will be made available on the vm's when the run starts.
This is the perfect way to add some small (think MB, not GB) files that are used as imput files. Typical scenarios are MD applications like Gromacs or QuantumESPRESSO (check out the examples). Again the output files are expected to be written in `outputs` to make sure the job picks them up after finishing.
