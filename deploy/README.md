# deploy

Deploy amlhpc infrastructure on Azure Machine Learning.

## Using the `deploy` command (amlhpc)

Initial deployment — workspace, dependencies, container registry, network, a login VM
and a default cluster (run from a checkout so the Bicep template is found):

```bash
deploy init -g amlhpc -l "centralus" -n amlhpc
deploy init -g amlhpc --what-if            # preview only, creates nothing
```

By default the login ComputeInstance has public SSH disabled (work from its
Jupyter/terminal, where the CI's managed identity authenticates automatically).
To enable public SSH — e.g. to drive the CI remotely for troubleshooting — pass a
public key. The CI still keeps its system-assigned identity and resource-group role,
so `sbatch`/`dask-*` authenticate the same way:

```bash
deploy init -g amlhpc -n amlhpc --enable-login-ssh --login-ssh-key ~/.ssh/id_rsa.pub
```

The key may be a `.pub` file path or the key string itself. Connect over the CI's
public IP on port 50000 as `azureuser` once provisioning completes.

Add a compute partition (AmlCompute cluster = Slurm partition) to an existing workspace.
Requires `SUBSCRIPTION`, `CI_RESOURCE_GROUP` and `CI_WORKSPACE` to be set:

```bash
deploy partition -n f4s -s Standard_F4s_v2 --max-nodes 5
deploy partition -n hb --size Standard_HB120rs_v3 --min-nodes 0 --max-nodes 8 --priority LowPriority
```

## Manual deployment (az CLI)

```bash
az group create --name amlhpc --location "Central US"
az deployment group create \
	--resource-group amlhpc \
	--template-file amlhpc_simple.bicep \
	--parameters name=amlhpc
```

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fhmeiland%2Famlhpc%2Fmaster%2Fdeploy%2Famlhpc_simple.json)
