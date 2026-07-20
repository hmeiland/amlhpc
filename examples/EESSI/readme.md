# EESSI

The EESSI project hosts a cvmfs based software stack, with scientific applications. For more info on the project visit: [https://eessi.io, https://eessi.io]

To access EESSI on your login vm, as root:
```
wget --no-check-certificate https://ecsft.cern.ch/dist/cvmfs/cvmfs-release/cvmfs-release-latest_all.deb && dpkg -i cvmfs-release-latest_all.deb
apt-get update && apt-get install -y cvmfs
wget https://github.com/EESSI/filesystem-layer/releases/download/v0.6.0/cvmfs-config-eessi_0.6.0_all.deb && dpkg -i cvmfs-config-eessi_0.6.0_all.deb
echo 'CVMFS_CLIENT_PROFILE="single"' > /etc/cvmfs/default.local
echo 'CVMFS_QUOTA_LIMIT=10000' >> /etc/cvmfs/default.local
echo 'CVMFS_REPOSITORIES=cms.cern.ch,pilot.eessi-hpc.org,software.eessi.io' >> /etc/cvmfs/default.local
echo 'CVMFS_HTTP_PROXY=DIRECT' >> /etc/cvmfs/default.local
cvmfs_config setup; mkdir -p /cvmfs/pilot.eessi-hpc.org /cvmfs/software.eessi.io
```

and then as the normal user:
```
sudo mount -t cvmfs software.eessi.io /cvmfs/software.eessi.io
source /cvmfs/software.eessi.io/versions/2023.06/init/bash
```

now you have access to all applications by simply loading the right modules. For an overview:
```
ml av
```

and for example OpenFOAM:
```
ml load OpenFOAM
source $FOAM_BASH
```


## Running EESSI applications as amlhpc jobs

The same cvmfs mount works inside the compute jobs themselves: the `amlhpc-ubuntu2004`
job container runs privileged with FUSE, so a job can mount EESSI, load a module and run
the application on the AmlCompute node. Submit the whole flow with `sbatch --wrap`:
```
sbatch -p f4s --wrap='sudo mount -t cvmfs software.eessi.io /cvmfs/software.eessi.io; \
source /cvmfs/software.eessi.io/versions/2023.06/init/bash; \
ml load OpenFOAM; source $FOAM_BASH; simpleFoam -help'
```
On the compute node `archdetect` picks the node's own micro-architecture (e.g.
`x86_64/intel/skylake_avx512` on an `f4s`), so the optimised build for that partition is used.


## Site-wide prolog/epilog: mount the stack once, per node

Wiring the cvmfs mount into every `--wrap` is repetitive. An admin can install a **site prolog**
(and optional epilog) once into the workspace storage stack; every `sbatch`/`srun` job then runs
it automatically before the user command, so the EESSI stack is already mounted and initialised
when the job starts.

Install the hooks from the login node (stored in `workspaceblobstore/amlhpc/{prolog,epilog}.sh`):
```
cat > prolog.sh <<'EOF'
# Mount EESSI on every node in the job (multi-node), or just this node for a
# single-node job. AZ_BATCH_HOST_LIST is only set when -N > 1.
if [ -n "$AZ_BATCH_HOST_LIST" ]; then
    parallel-ssh -i -H "${AZ_BATCH_HOST_LIST//,/ }" \
        "sudo mount -t cvmfs software.eessi.io /cvmfs/software.eessi.io"
else
    sudo mount -t cvmfs software.eessi.io /cvmfs/software.eessi.io
fi
source /cvmfs/software.eessi.io/versions/2023.06/init/bash
EOF
deploy config set-prolog prolog.sh
deploy config show
```
Users then load their application and run it in the `--wrap` (or runscript):
```
sbatch -p f4s --wrap='ml load OpenFOAM; source $FOAM_BASH; simpleFoam -help'
```

**The prolog deliberately does not `ml load` any application.** Its job is to make the software
stack *available* — mount cvmfs (on every node, via `parallel-ssh`, so multi-node MPI jobs work
too) and initialise EESSI. **Choosing which application to load is the user's responsibility**
and belongs in the job command: a site prolog is shared by *every* job on the workspace, so a
hard-coded `ml load OpenFOAM` there would force OpenFOAM onto unrelated GROMACS, Quantum ESPRESSO
or OSU-benchmark jobs. Keeping the split at "prolog mounts, job loads" lets one prolog serve
every application while each job stays in control of its own environment.

Every job is wrapped as `prolog → ( user command ) → epilog`; the user command runs in a subshell
so a failing/`exit`-ing command still runs the epilog and its exit code is preserved. Pass
`--no-prolog` to `sbatch`/`srun` to opt a single job out. Remove with `deploy config clear-prolog`.

alternativly, you can also use the same docker container as through which the jobs are running:
```
docker run -ti --privileged -v `pwd`:/app  docker.io/hmeiland/amlhpc-ubuntu2004 bash
azureuser@126592f454be:/app$ sudo mount -t cvmfs software.eessi.io /cvmfs/software.eessi.io
CernVM-FS: running with credentials 109:110
CernVM-FS: loading Fuse module... done
CernVM-FS: mounted cvmfs on /cvmfs/software.eessi.io
azureuser@126592f454be:/app$ source /cvmfs/software.eessi.io/versions/2023.06/init/bash 
Found EESSI repo @ /cvmfs/software.eessi.io/versions/2023.06!
archdetect says x86_64/intel/skylake_avx512
Using x86_64/intel/skylake_avx512 as software subdirectory.
Using /cvmfs/software.eessi.io/versions/2023.06/software/linux/x86_64/intel/skylake_avx512/modules/all as the directory to be added to MODULEPATH.
Found Lmod configuration file at /cvmfs/software.eessi.io/versions/2023.06/software/linux/x86_64/intel/skylake_avx512/.lmod/lmodrc.lua
Initializing Lmod...
Prepending /cvmfs/software.eessi.io/versions/2023.06/software/linux/x86_64/intel/skylake_avx512/modules/all to $MODULEPATH...
Environment set up to use EESSI (2023.06), have fun!
{EESSI 2023.06} azureuser@126592f454be:/app$ ml load OpenFOAM
{EESSI 2023.06} azureuser@126592f454be:/app$ source $FOAM_BASH
```
