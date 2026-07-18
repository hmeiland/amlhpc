# amlhpc

**A -just enough- Slurm / PBS / LSF experience on Azure Machine Learning.**

Use the familiar `sbatch` / `qsub` / `bsub` / `sinfo` / `squeue` commands to submit jobs and
inspect the state of your HPC system on Azure Machine Learning — without re-programming another
integration.

## Installation

```bash
pip install amlhpc
```

## Configuration

The commands require the following environment variables:

```bash
export SUBSCRIPTION=<guid of your Azure subscription>
export CI_RESOURCE_GROUP=<resource group of your Azure ML Workspace>
export CI_WORKSPACE=<name of your Azure ML Workspace>
```

Inside the Azure Machine Learning environment, `CI_RESOURCE_GROUP` and `CI_WORKSPACE` are
normally already set, so you only need to export `SUBSCRIPTION`.

Every command is available both as a standalone executable (`sbatch`, `srun`, `sinfo`, ...) and
as a subcommand of the unified `amlhpc` entry point, so `amlhpc sbatch ...` is equivalent to
`sbatch ...`.

```{toctree}
:hidden:
:maxdepth: 2
:caption: Contents

commands
data
dask
architecture
examples
```
