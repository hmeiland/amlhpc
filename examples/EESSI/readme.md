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
ml load OpenFOAM
source $FOAM_BASH
```
