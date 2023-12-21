# Data handling

## sbatch --wrap "hostname"

For the most simple command-line job, not data is transferred from the submit host to the cluster. 
The wrapped command is forwarded and the standard-out is captured in the job output std_log.txt file which can be found in the "Outputs and log" tab of the job.
Any files created in the `outputs` folder of the work-dir are also captured by the job and can be found in the same tab.

