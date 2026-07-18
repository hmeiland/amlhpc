# Architecture

How amlhpc integrates with Azure Machine Learning (AML).

amlhpc is a thin client. It ships no scheduler, no daemon and no server of its own: every
command translates a familiar HPC gesture (`sbatch`, `qsub`, `bsub`, `sinfo`, `squeue`) into
one or more calls against the AML control plane through the `azure-ai-ml` SDK, then formats
the response back into scheduler-style output. AML *is* the scheduler, the job store and the
compute manager; amlhpc is the muscle memory on top of it.

```
 user shell                amlhpc (client)                      Azure Machine Learning
------------           ----------------------           -----------------------------------
 sbatch -p f4s ...  ->  build CommandJob / SweepJob  ->  jobs.create_or_update  ->  AmlCompute cluster
 srun --wrap ...    ->  build CommandJob             ->  jobs.create_or_update  ->  ComputeInstance
 sinfo              ->  compute.list()               ->  workspace computes
 squeue / qstat     ->  jobs.list()                  ->  workspace job history
 scancel / qdel     ->  jobs.begin_cancel(id)        ->  workspace job
 deploy init        ->  az deployment group create   ->  Bicep template (whole workspace)
 deploy partition   ->  compute.begin_create_or_update -> new AmlCompute cluster
 deploy config      ->  datastores.get + blob upload ->  workspaceblobstore/amlhpc/*.sh
```

## The command surface

Every command is both a standalone entry point and a subcommand of the unified `amlhpc`
launcher; the `[project.scripts]` table in `pyproject.toml` wires each name to a function, and
[`amlhpc/__main__.py`](amlhpc/__main__.py) dispatches `amlhpc <command>` to the same functions.
So `sbatch ...` and `amlhpc sbatch ...` are identical.

The commands are grouped by the scheduler dialect they imitate, but they all target the same AML
primitives:

| Dialect | Commands | AML operation |
| --- | --- | --- |
| Slurm | `sbatch`, `srun`, `sinfo`, `squeue`, `scancel` | jobs + computes |
| PBS | `qsub`, `qstat`, `qdel` | jobs (qsub is a front-end for sbatch) |
| LSF | `bsub`, `bjobs`, `bkill` | jobs (bsub is a front-end for sbatch) |

`qsub`/`bsub` do not talk to AML directly — they parse PBS/LSF-style flags and re-invoke
`sbatch` (see [`amlhpc/pbs/qsub.py`](amlhpc/pbs/qsub.py)): `-q`→`-p`, `-l nodes=N`/`select=N`→`-N`,
`-N`/`-J` job name is informational (AML assigns the JOBID). `qstat`/`bjobs` and `qdel`/`bkill`
share one implementation in [`amlhpc/jobcontrol.py`](amlhpc/jobcontrol.py).

## Authentication

Each command builds an `MLClient` from three environment variables — `SUBSCRIPTION`,
`CI_RESOURCE_GROUP`, `CI_WORKSPACE` — and picks a credential based on where it runs:

- **On an AML ComputeInstance** (`APPSETTING_WEBSITE_SITE_NAME == "AMLComputeInstance"`) it uses
  the CI's managed identity via the local MSI endpoint (`mlComputeAuth`), so no interactive login
  is needed from the login node.
- **Everywhere else** it falls back to `DefaultAzureCredential` (az CLI login, env, VS Code, etc.).

This pattern is duplicated across the command modules rather than centralised, so each command is
self-contained. The `deploy config` path uses the same logic (`config_ml_client` in
[`amlhpc/deploy.py`](amlhpc/deploy.py)).

## Submitting a job (`sbatch`)

[`amlhpc/slurm/sbatch.py`](amlhpc/slurm/sbatch.py) is the core translator. It maps HPC concepts
onto an AML job as follows:

- **partition (`-p`) → `compute`**: the partition name is the AmlCompute cluster name. `sinfo`
  lists them, `deploy partition` creates them.
- **nodes (`-N`) → `instance_count`**: multi-node jobs set `instance_count` and export
  `SLURM_JOB_NODES`.
- **`--wrap "cmd"`**: no code is uploaded (`code=None`); the command string is the job command.
- **`script`**: the single script file is uploaded as the job `code` and executed
  (`chmod +x; ./script`).
- **environment / container**: `-e` selects a named AML environment; `--container` registers a
  Docker image as an ad-hoc environment. The default is `amlhpc-ubuntu2204@latest`, the image the
  Bicep template registers at deploy time (`docker.io/hmeiland/amlhpc-ubuntu2204`).
- **`--array` → AML SweepJob**: array jobs become a `SweepJob` sweeping `SLURM_ARRAY_TASK_ID` over
  the requested index set, with `max_concurrent_trials` bounding parallelism — AML's sweep engine
  stands in for the array scheduler.
- **`--datamover`** stages input/output data (see [data.md](data.md)): `simple` uploads the working
  directory as job code; `datastore` mounts a matching AML datastore as the job workdir via an
  `Output` URI; `nfs` mounts an NFS export discovered from the submit host's mount table.

Whichever path is taken, the job is realised with a single `ml_client.jobs.create_or_update(...)`
and the returned job name (the AML-assigned JOBID) is printed — the same string `squeue`/`qstat`
list and `scancel`/`qdel` cancel.

## Running on the login node (`srun`)

[`amlhpc/slurm/srun.py`](amlhpc/slurm/srun.py) mirrors `sbatch` but targets a ComputeInstance
instead of a cluster. With no `-p` it auto-discovers the single ComputeInstance in the workspace
(`compute.list()` filtered to `computeinstance`). It is used for control-plane tasks that must run
on the login node itself — notably bringing up a Dask scheduler — without needing SSH. Like every
AML command job, the command still runs inside the environment's container sharing the CI's network
namespace, not on the bare CI host OS.

## Site-wide prolog/epilog (`deploy config`)

amlhpc keeps optional site-wide *prolog* and *epilog* shell snippets in the workspace's default
blob datastore (`workspaceblobstore`, prefix `amlhpc/`), so common boilerplate (mounting a shared
software stack, loading a module, sourcing an environment) lives once in the storage stack instead
of in every job. See [`amlhpc/config.py`](amlhpc/config.py).

- **Storage as source of truth**: the hooks are plain blobs at
  `workspaceblobstore/amlhpc/{prolog,epilog}.sh`. `deploy config set-prolog/set-epilog/show/clear-*`
  reads and writes them. Access uses the datastore's own stored credentials (SAS or account key)
  from `ml_client.datastores.get(..., include_secrets=True)`, so the plain Contributor role that
  `deploy` grants is enough — no separate data-plane blob RBAC.
- **Automatic wrapping**: on every submission `sbatch`/`srun` call `apply_site_hooks`, which loads
  the hooks and wraps the job command as `prolog → ( user command ) → epilog`. The user command runs
  in a subshell so a failing or `exit`-ing command still runs the epilog, and its real exit code is
  preserved and re-raised after the epilog. `--no-prolog` opts a single job out. Missing hooks or an
  unreadable datastore degrade to "no hook" rather than blocking submission.
- **Profile as pointer**: the `~/.amlhpc/<name>.sh` profile that `deploy init` writes records
  `AMLHPC_CONFIG_DATASTORE` / `AMLHPC_CONFIG_PREFIX`, letting the client find the config. The blob is
  the source of truth; the profile only points at it.

A worked example (EESSI software stack) is in [examples/EESSI/readme.md](examples/EESSI/readme.md).

## Provisioning (`deploy`)

[`amlhpc/deploy.py`](amlhpc/deploy.py) is the one place amlhpc reaches past the AML SDK:

- **`deploy init`** shells out to `az deployment group create` with the bundled Bicep template
  ([`amlhpc/templates/amlhpc_simple.bicep`](amlhpc/templates/amlhpc_simple.bicep)), which stands up
  the full workspace: storage account, key vault, Log Analytics + Application Insights, a container
  registry, the AML workspace, a VNet with cluster/ANF subnets, a login ComputeInstance, a default
  AmlCompute cluster, a resource-group role assignment for the CI identity, and the default
  `amlhpc-ubuntu2204` environment. On success it writes the `~/.amlhpc/<name>.sh` profile.
- **`deploy partition`** adds an AmlCompute cluster (a Slurm "partition") to an existing workspace
  via `compute.begin_create_or_update`, with size / min-max nodes / idle time / priority.
- **`deploy config`** manages the site prolog/epilog described above.

## Dask integration

The `dask-scheduler-up` / `dask-up` / `dask-down` commands compose the primitives above rather than
adding new AML integration. The scheduler ([`amlhpc/dask/scheduler.py`](amlhpc/dask/scheduler.py))
runs on the login ComputeInstance (typically launched through `srun`), binding to the CI's private
IP (auto-detected via the Azure Instance Metadata Service) so that AmlCompute workers on the shared
VNet can reach it. Workers are then submitted as ordinary jobs pointed at the advertised
`tcp://<private-ip>:8786` scheduler address. The VNet created by `deploy init` is what makes this
private-IP routing between the CI and the clusters work.

## Design notes

- **Stateless client**: amlhpc stores nothing server-side of its own. JOBIDs, job state and history
  all live in AML; `squeue`/`qstat` are just formatted views of `jobs.list()`. This is why the tools
  work interchangeably from a laptop (`DefaultAzureCredential`) or the login CI (managed identity).
- **Job name = JOBID**: AML assigns the job name; amlhpc surfaces it verbatim as the JOBID so the
  cancel/query commands line up with what submission printed.
- **"Just enough"**: the goal is familiar ergonomics over a faithful reimplementation. Flags that do
  not map onto an AML concept (e.g. PBS/LSF job names) are accepted and treated as informational
  rather than silently rejected.
