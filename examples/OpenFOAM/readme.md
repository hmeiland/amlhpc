# OpenFOAM

This example runs the motorbike tutorial, which defaults to 8 domains.
The runscript does do some nasty stuff to allow the example to be copied out of the read-only filesystem.


```
sudo mount -t cvmfs software.eessi.io /cvmfs/software.eessi.io
source /cvmfs/software.eessi.io/versions/2023.06/init/bash
ml load OpenFOAM
source $FOAM_BASH

mkdir outputs
cd outputs

# Brute force copy, since this comes from a read-only, non-owned filesystem 
sudo cp -r $FOAM_TUTORIALS/incompressibleFluid/motorBike .
sudo chown -R `id -u`:`id -g` motorBike
sudo chmod -R 777 motorBike


cd motorBike
./Allrun
```
