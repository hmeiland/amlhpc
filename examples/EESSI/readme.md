# EESSI

The EESSI project hosts a cvmfs based software stack, with scientific applications. For more info on the project visit: [https://eessi.io, https://eessi.io]

To access EESSI on your login vm, as root:
```
wget --no-check-certificate https://ecsft.cern.ch/dist/cvmfs/cvmfs-release/cvmfs-release-latest_all.deb && dpkg -i cvmfs-release-latest_all.deb
apt-get update && apt-get install -y cvmfs
wget https://github.com/EESSI/filesystem-layer/releases/download/v0.5.0/cvmfs-config-eessi_0.5.0_all.deb && dpkg -i cvmfs-config-eessi_0.5.0_all.deb
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
