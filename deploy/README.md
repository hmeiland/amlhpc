# deploy

Deploy amlhpc infrastructure on Azure Machine Learning.

## Using the `deploy` command (amlhpc)

Initial deployment — workspace, dependencies, container registry, network, a login VM
and a default cluster. `deploy` is also available as `amlhpc deploy`. The Bicep template
ships with the package, so this works from any directory once amlhpc is pip-installed
(pass `--template` to override with your own):

```bash
deploy init -g amlhpc -l "centralus" -n amlhpc
amlhpc deploy init -g amlhpc --what-if      # preview only, creates nothing
```

On a successful `init` (not `--what-if`), a cluster profile is written to
`~/.amlhpc/<name>.sh` holding the `SUBSCRIPTION`, `CI_RESOURCE_GROUP` and
`CI_WORKSPACE` exports for the new workspace. Coming back to the cluster later,
source it to restore the environment `sbatch`/`dask-*`/`deploy partition` need:

```bash
source ~/.amlhpc/amlhpc.sh
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

Manage the site-wide prolog/epilog stored in the workspace storage stack. These hooks are
run by every `sbatch`/`srun` job (as `prolog → ( user command ) → epilog`), so users can
submit bare application commands. They live in the workspace's default datastore at
`workspaceblobstore/amlhpc/{prolog,epilog}.sh` (override with `AMLHPC_CONFIG_DATASTORE` /
`AMLHPC_CONFIG_PREFIX`). Requires `SUBSCRIPTION`, `CI_RESOURCE_GROUP` and `CI_WORKSPACE`:

```bash
deploy config set-prolog prolog.sh    # upload a shell snippet as the site prolog
deploy config set-epilog epilog.sh    # upload a shell snippet as the site epilog
deploy config show                    # print the current prolog and epilog
deploy config clear-prolog            # remove the site prolog
deploy config clear-epilog            # remove the site epilog
```

Individual jobs opt out with `sbatch --no-prolog` / `srun --no-prolog`. A worked EESSI
example is in [examples/EESSI/readme.md](../examples/EESSI/readme.md).

## Manual deployment (az CLI)

```bash
az group create --name amlhpc --location "Central US"
az deployment group create \
	--resource-group amlhpc \
	--template-file amlhpc_simple.bicep \
	--parameters name=amlhpc
```

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fhmeiland%2Famlhpc%2Fmaster%2Fdeploy%2Famlhpc_simple.json)
