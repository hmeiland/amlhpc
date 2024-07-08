Azure Machine learning can use a few compute back-ends. While AMLHPC typically uses the Batch based Compute clusters, the downside of this is the requirement for AML specific quota. Another option is to use Kubernetes clusters, for which AKS is also a valid target. Here you can use your normal subscription quota.

Here is a quick guide for setting up a small AKS cluster and connecting it to Azure Machine Learning:
```
# create resource group
az group create --name demo-aks --location westeurope

# create AKS, with managed identity and specific VM type 
az aks create -g demo-aks -n aml-aks --enable-managed-identity -s Standard_D16as_v5

# install the AML extension to allow AML to use the AKS cluster
az k8s-extension create --name amlaksext --extension-type Microsoft.AzureML.Kubernetes --config enableTraining=True enableInference=True inferenceRouterServiceType=LoadBalancer allowInsecureConnections=True InferenceRouterHA=False --cluster-type managedClusters --cluster-name aml-aks --resource-group demo-aks --scope cluster

# check the Container Registry name and make sure the aks is registrered for using it
az aks update --name aml-aks --resource-group demo-aks --attach-acr <aml container registry>
```

Once the AKS is created, you can attach it to AML. To do this, go to the Kubernetes Clusters tab in Compute; press new and select Kubernetes. Provide a name that will be used as the partition name and select the AKS from the drop-down list.

From here on you can submit a job from the command line e.g.:
```
sbatch -p aks --wrap="hostname"
```


To make use of the huge EESSI software repository, some additional steps are required:
```
# get credentials for kubectl and helm to e.g. installing eessi
az aks get-credentials --resource-group demo-aks --name aml-aks

# download the script to mirror and load the csi containers into your AKS
wget https://raw.githubusercontent.com/hmeiland/amlhpc/main/deploy/aks-eessi/mirror_containers_and_install_csi.sh

# run the script
./mirror_containers_and_install_csi.sh <aml container registry>

# download the pvc config
wget https://raw.githubusercontent.com/hmeiland/amlhpc/main/deploy/aks-eessi/aml-cvmfs-pvc.yaml

# load the pvc config to automount the /cvmfs in the AML containers
kubectl create -f aml-cvmfs-pvc.yaml
```

To verify it works:
```
sbatch -p aks --wrap="source /cvmfs/software.eessi.io/versions/2023.06/init/bash"
```

the job output should be:
```
Found EESSI repo @ /cvmfs/software.eessi.io/versions/2023.06!
archdetect says x86_64/amd/zen3
Using x86_64/amd/zen3 as software subdirectory.
Found Lmod configuration file at /cvmfs/software.eessi.io/versions/2023.06/software/linux/x86_64/amd/zen3/.lmod/lmodrc.lua
Found Lmod SitePackage.lua file at /cvmfs/software.eessi.io/versions/2023.06/software/linux/x86_64/amd/zen3/.lmod/SitePackage.lua
Using /cvmfs/software.eessi.io/versions/2023.06/software/linux/x86_64/amd/zen3/modules/all as the directory to be added to MODULEPATH.
Using /cvmfs/software.eessi.io/host_injections/2023.06/software/linux/x86_64/amd/zen3/modules/all as the site extension directory to be added to MODULEPATH.
Initializing Lmod...
Prepending /cvmfs/software.eessi.io/versions/2023.06/software/linux/x86_64/amd/zen3/modules/all to $MODULEPATH...
Prepending site path /cvmfs/software.eessi.io/host_injections/2023.06/software/linux/x86_64/amd/zen3/modules/all to $MODULEPATH...
Environment set up to use EESSI (2023.06), have fun!
```

Have fun!


There are still a few thinks to figure out:

- job scripts give the error `/bin/bash: ./runscript.sh: Permission denied`
