Azure Machine learning can use e few compute back-ends. While AMLHPC typically uses the Batch based Compute clusters, the downside of this is the requirement for AML specific quota. Another option is to use Kubernetes clusters, for which AKS is also a valid target. Here you can use your normal subscription quota.

Here is a quick guide for setting up a small AKS cluster and connecting it to Azure Machine Learning:
```
# create resource group
az group create --name demo-aks --location westeurope

# create AKS, with managed identity and specific VM type 
az aks create -g demo-aks -n aml-aks --enable-managed-identity -s Standard_D16as_v5

# install the AML extension to allow AML to use the AKS cluster
az k8s-extension create --name amlaksext --extension-type Microsoft.AzureML.Kubernetes --config enableTraining=True enableInference=True inferenceRouterServiceType=LoadBalancer allowInsecureConnections=True InferenceRouterHA=False --cluster-type managedClusters --cluster-name aml-aks --resource-group demo-aks --scope cluster

# get credentials for e.g. installing eessi
az aks get-credentials --resource-group demo-aks --name aml-aks

# make sure the aks is registrered for the AML Container registry
az aks update --name aml-aks --resource-group demo-aks --attach-acr cramlhpcy6smovsa2coz
```

Once the AKS is created, you can attach it to AML. To do this, go to the Kubernetes Clusters tab in Compute; press new and select Kubernetes. Provide a name that will be used as the partition name and select the AKS from the drop-down list.

From here on you can submit a job from the command line e.g.:
```
sbatch -p aks --wrap="hostname"
```


There are still a few thinks to figure out:

-- job scripts give the error `/bin/bash: ./runscript.sh: Permission denied`
-- unable to mount cvmfs, there needs to be a csi driver on Kubernetes
