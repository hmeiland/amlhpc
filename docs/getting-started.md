# Getting started

This walkthrough takes you from a fresh install to your first job running on Azure
Machine Learning. It should take about ten minutes plus provisioning time.

## Prerequisites

- **Python 3.10 or newer.**
- **The Azure CLI (`az`)**, logged in to the subscription that holds (or will hold) your
  workspace:
  ```bash
  az login
  az account set --subscription <subscription-guid>
  ```
- **Contributor / owner rights** on a resource group if you want `amlhpc` to provision a new
  workspace for you. To use an *existing* workspace you only need access to it — no admin
  rights required.

## 1. Install

```bash
pip install amlhpc
```

This installs the unified `amlhpc` entry point plus the familiar standalone commands
(`sbatch`, `srun`, `sinfo`, `squeue`, `deploy`, ...). `amlhpc sbatch ...` and `sbatch ...`
are equivalent.

## 2. Point amlhpc at a workspace

Every command needs to know which Azure Machine Learning workspace to talk to — a
subscription, a resource group and a workspace name. Pick the path that matches your
situation.

### Path A — I already have a workspace

Register it once as a **cluster profile**. This writes `~/.amlhpc/config.json` (an
OS-neutral file, no shell exports to juggle) and makes the workspace your *current* cluster:

```bash
deploy connect -n prod -s <subscription> -g <resource-group> -w <workspace>
```

Subscription defaults to `$SUBSCRIPTION` if you omit `-s`. The profile only *names* the
workspace — it holds no credentials, so every command still authenticates as your own Azure
identity, and profiles are safe to share.

### Path B — I want amlhpc to build one for me

`deploy init` provisions a complete environment: the workspace, its dependencies, a
container registry, networking, a login ComputeInstance and a default compute cluster.

```bash
deploy init -g amlhpc -l "centralus" -n amlhpc --what-if   # preview — creates nothing
deploy init -g amlhpc -l "centralus" -n amlhpc             # actually provision
```

On success the new workspace is registered as a cluster profile and made current
automatically, so the commands below work with no further setup.

### Path C — I just want environment variables

The legacy environment variables are fully supported and take precedence over the current
profile, so nothing that relies on them breaks:

```bash
export SUBSCRIPTION=<subscription-guid>
export CI_RESOURCE_GROUP=<resource-group>
export CI_WORKSPACE=<workspace-name>
```

Inside the Azure Machine Learning environment, `CI_RESOURCE_GROUP` and `CI_WORKSPACE` are
normally already set, so you only need to export `SUBSCRIPTION`.

## 3. Verify the setup

Confirm the workspace has everything amlhpc needs — a default environment, at least one
compute partition, a login ComputeInstance and the `workspaceblobstore` datastore:

```bash
deploy doctor            # report readiness of the current cluster
deploy doctor --fix      # additionally create the default environment if it is missing
```

If you registered more than one workspace, `amlhpc clusters` lists your profiles (`*` marks
the current one) and `amlhpc use <name>` switches between them.

## 4. See what compute is available

```bash
sinfo
```

```text
PARTITION       AVAIL   VM_SIZE                 NODES   STATE
f16s            UP      STANDARD_F16S_V2        37
hbv2            UP      STANDARD_HB120RS_V2     4
login-vm        UP      STANDARD_DS12_V2        None
```

The `PARTITION` names are what you pass to `-p` when submitting a job.

## 5. Submit your first job

Run a command on a compute partition with `sbatch --wrap`:

```bash
sbatch -p f16s --wrap="hostname"
```

```text
gifted_engine_yq801rygm2
```

The returned string is the **JOBID**. You can also submit a script file instead of `--wrap`,
and PBS (`qsub`) and LSF (`bsub`) front-ends are available if that muscle memory is stronger.

## 6. Watch it and get the output

```bash
squeue                       # list jobs and their STATE
sacct gifted_engine_yq801rygm2   # status + start/end time for one job
sattach gifted_engine_yq801rygm2 # print the job's log (add -f to follow live)
```

Cancel a job with `scancel <JOBID>` (or `qdel` / `bkill`).

## Next steps

- **[Commands](commands.md)** — the full reference for every `sbatch` / `squeue` / `sstat`
  option and the PBS/LSF equivalents.
- **[Data handling](data.md)** — moving input and output data along with your jobs.
- **[Dask on amlhpc](dask.md)** — stand up a distributed Dask cluster.
- **[Examples](examples.md)** — end-to-end walkthroughs for OpenFOAM, GROMACS, TensorFlow,
  EESSI and more.
- **[Architecture](architecture.md)** — how amlhpc maps the Slurm/PBS surface onto Azure
  Machine Learning.
