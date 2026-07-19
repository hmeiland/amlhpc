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

On a successful `init` (not `--what-if`), the new workspace is registered as a cluster profile
in `~/.amlhpc/config.json` and made the current cluster, so `sbatch`/`dask-*`/`deploy partition`
work immediately with no exports (see [Cluster profiles](#cluster-profiles) below). For
backward compatibility a `~/.amlhpc/<name>.sh` with the `SUBSCRIPTION`, `CI_RESOURCE_GROUP` and
`CI_WORKSPACE` exports is also written; coming back later you can either rely on the stored
profile or source the shell file:

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

## Cluster profiles

amlhpc records which Azure Machine Learning workspace to talk to as a **cluster profile** in
`~/.amlhpc/config.json` — an OS-neutral file (stored `0600`) read directly on Linux, Windows and
inside AML, so there is no shell file to `source` and no per-shell exports to juggle. A profile
holds only the three identifiers that name a workspace (subscription, resource group, workspace);
it is **not secret** and carries no credentials — every command still authenticates as the
caller's own Azure identity. `resolve_connection` applies this precedence so nothing that works
today breaks: `--cluster NAME` → `AMLHPC_CLUSTER` env → legacy `SUBSCRIPTION`/`CI_RESOURCE_GROUP`/`CI_WORKSPACE`
env → the `current` profile in the config file.

Register an existing workspace (the non-provisioning counterpart to `init`) and switch between
clusters. Subscription defaults to `$SUBSCRIPTION` when `-s` is omitted:

```bash
deploy connect -n prod -s <subscription> -g <resource-group> -w <workspace>
deploy connect -n dev  -g <resource-group> -w <workspace> --no-current   # register without making current
deploy connect -n prod -g <rg> -w <ws> --check                            # register then run doctor
amlhpc clusters          # list profiles ( * marks the current one)
amlhpc use dev           # switch the current cluster
```

Check that a workspace has everything the amlhpc commands need (default environment, at least
one AmlCompute partition, a login ComputeInstance, the `workspaceblobstore` datastore). `--fix`
creates the missing default environment:

```bash
deploy doctor            # report readiness of the current cluster
deploy doctor --cluster prod --fix   # check a named cluster and create the default env if absent
```

Share a profile with a teammate and import it on another machine. The shared pointer carries only
the four identifier fields — no credentials, no current-pointer, no other clusters:

```bash
deploy share prod -o prod.json       # write a portable pointer (omit -o to print to stdout)
deploy import -f prod.json           # merge it locally and make it current
deploy import -f prod.json --no-current --check   # import without switching, then run doctor
```

A shared profile confers no access on its own. Grant a teammate the workspace-scoped **AzureML
Data Scientist** role (or revoke it) via the Azure CLI. If `az` is not installed or you lack
permission to manage role assignments, the exact `az role assignment` command is printed for an
admin to run:

```bash
deploy invite alice@example.com                 # grant access to the current cluster's workspace
deploy invite alice@example.com --cluster prod -y   # named cluster, skip the confirm prompt
deploy uninvite alice@example.com               # revoke access
```

## deploy partition / config

Add a compute partition (AmlCompute cluster = Slurm partition) to an existing workspace.
Requires a resolvable cluster (a profile or the legacy env vars):

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
