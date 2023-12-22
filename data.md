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

## sbatch --datamover=datastore ./runscript

On your Compute vm, you have the option to mount a datastore, which in itself can be managed from the Data - Datastores in AML Studio. This datastore will be mounted as ~/cloudfiles/data/<datastore>. Amlslurm can use this datastore as a "shared filesystem", and can also mount the datastore on the compute vm's for the job. Make sure to name the mountpoint the same as the datastore name to allow amlslurm to recognize it. The mount will be made at the present work directory level, so the job has no access to higher level directories and files. On the compute side, the filelist does not get refreshed, thus you have to unmount and again mount to see the result files.

The datastore option will move up in the PWD path to find a mountpoint with the same name as a datastore. This datastore together with the relative PWD is used to mount the datastore in the Compute Cluster vm's.

## sbatch --datamover=nfs ./runscript.sh

When both the Compute vm's and the Compute Clusters are created with the same Vnet, they can all mount any NFS services that are available. When submitting a job, the PWD is travelled to find a mountpoint, which is then checked against available nfs mountpoints. The server and mountpoint are provided to the job and on startup of the Compute Cluster vm's these are mounted. On start of the job, the working directory will be set to the PWD during the job submission. 
