# environments

Container environments for amlhpc jobs on Azure Machine Learning. Each
subdirectory is a build context containing a `Dockerfile`.

Build and register them into your AML workspace with `container`, which uses
Azure ML's native build workflow and provisions the workspace container
registry if one does not exist yet:

```bash
container                       # build every environment in this directory
container -e amlhpc-ubuntu2204  # build a single environment
```

`amlhpc-ubuntu2204` is the default job environment used by `sbatch` (referenced
as `amlhpc-ubuntu2204@latest`).
