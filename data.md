# Data handling

## sbatch --wrap "hostname"

For the most simple command-line job, not data is transferred from the submit host to the cluster. 
The wrapped command is forwarded and the standard-out is captured in the job output `std_log.txt` file which can be found in the `Outputs and log` tab of the job.
Any files created in the `./outputs` folder of the workdir are also captured by the job and can be found in the same tab.

## sbatch ./runscript.sh

For a job which is described in a `runscript.sh` (or any other filename), the `runscript.sh` (and only the runscript.sh) is copied into the job and can be found in the `Code` tab of the job. 
During the job preparation it is copied into the workdir, and since the command points to this script, it will be executed. 
Any files created in the `./outputs` folder of the workdir are also captured by the job and can be found in the same tab.
