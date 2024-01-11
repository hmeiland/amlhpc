
Deployment steps of AML HPC cluster:

```
az group create --name amlhpc --location "Central US"
az deployment group create \
	--resource-group amlhpc \
   	--template-file amlhpc_simple.bicep \
	--parameters name=amlhpc location=centralus
```

https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fhmeiland%2Famlhpc%2Fmaster%2Fdeploy%2Famlhpc_simple.json
