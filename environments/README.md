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

## Why we rebuild OpenMPI (InfiniBand / UCX)

The HB-series VMs (`HB120rs_v3`, `HB176rs_v4`, ...) carry a Mellanox/NVIDIA
**InfiniBand** fabric. To actually use it, MPI has to go through **UCX**
(OpenMPI's `pml_ucx`) over the mlx5 verbs devices. The stock AzureML base
images do **not** ship that, so every Dockerfile here that runs MPI rebuilds
OpenMPI from source `--with-ucx`. Three findings drove this, each verified on
real Azure:

1. **The stock base is TCP-only.** `mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04`
   (and the newer `openmpi5.0-ubuntu24.04`, OpenMPI 5.0.6) ship OpenMPI with
   **no UCX and no verbs** — `ompi_info` shows no `pml: ucx` (only `cm/ob1/
   monitoring/v`), and there is no `libucp`/`libibverbs`/`librdmacm`/`libmlx5`.
   MPI silently falls back to the
   TCP btl over Ethernet, so bandwidth caps at single-digit Gb/s instead of the
   ~189 Gb/s the 200 Gb/s HDR fabric can deliver. No published AzureML base
   image ships InfiniBand support; the Azure *HPC VM* images do, but those are
   a different product and bypass the AzureML job launcher.

2. **`orted` must stay at `/usr/local/openmpi`.** A multi-node (`-N > 1`) job
   runs under an AML `MpiDistribution`, whose launcher bridges `orted` between
   nodes **at job startup**. That bridge only works when `mpirun`/`orted`
   resolve to `/usr/local/openmpi` (where the stock base puts them); an MPI
   installed anywhere else (e.g. `/usr/local/bin`) makes the leader hang ~30s
   on `remote-executor` and the job never starts. So we rebuild the **same**
   OpenMPI **in place** at `/usr/local/openmpi` rather than installing a second
   copy — this also keeps one ABI-consistent runtime under AML's launcher.

   This is also why we can't just symlink `/usr/local/openmpi` at an EESSI
   OpenMPI: EESSI only mounts (via CVMFS) *inside* the runscript, which runs
   **after** the launcher already needs `orted`, so the symlink dangles at the
   critical moment; and EESSI's OpenMPI (4.1.6, built against EESSI's own
   glibc/GCC compat layer) is ABI-skewed from AML's launcher runtime.

3. **The container must run as root.** The launcher's cross-node bridge is a
   Unix-domain gRPC socket under `/mnt/azureml/cr/j/.../` that is reachable only
   by root. A `USER azureuser` (uid 1000) directive causes the same ~30s
   `remote-executor` timeout. Keep the `azureuser` account if tooling needs the
   home directory, but do **not** switch to it — run as root.

4. **Stay on the OpenMPI 4.1.0 (ORTE) base — the OpenMPI 5.0.6 base is not a
   usable IB path.** The newer `openmpi5.0-ubuntu24.04` base ships OpenMPI 5.0.6,
   which replaced the legacy **ORTE** runtime (`orted`) with **PRRTE + PMIx**
   (`prted`). AML's multi-node launcher (`common_runtime`) drives that daemon
   itself, constructing a detailed `prted` bootstrap command (`PRTE_PREFIX`,
   `--prtemca ess/PREFIXES/prte_hnp_uri/plm ssh ...`) that is **version-locked**
   to the exact PRRTE/PMIx the base was built with (PRRTE 3.0.7 / PMIx 2.13.x).
   Rebuilding OpenMPI 5.0.6 from the vanilla source tarball — as we must, to add
   UCX — also swaps that PRRTE/PMIx **control plane** for the tarball's own
   internal copies, and the launcher then aborts the daemon at startup with
   `PMIX_ERR_BAD_PARAM` in `gds_utils.c`/`gds_hash.c` (verified on a real 2-node
   hbv3 job). OpenMPI 4.1.0/ORTE does **not** have this coupling: ORTE is
   self-contained, so the launcher only needs a compatible `orted` at
   `/usr/local/openmpi` and the `--with-ucx` rebuild changes only the data plane.

   The non-invasive alternative fails too: adding just `libucx-dev` to the stock
   OpenMPI 5.0.6 does **not** enable UCX, because that OpenMPI was compiled
   `--without-ucx` — `ompi_info` lists only `pml: cm/ob1/monitoring/v` and there
   is no `mca_pml_ucx.so` on disk, so no runtime MCA var can activate it. Getting
   UCX into OpenMPI 5.x on AML would require rebuilding it against the *exact*
   AzureML-shipped PRRTE/PMIx (not just a matching upstream version), which the
   base does not provide the sources/headers for. Until that changes, **the
   OpenMPI 4.1.0 ORTE base is the supported InfiniBand path.**

Practical recipe (see [`openmpi-eessi/Dockerfile`](openmpi-eessi/Dockerfile) for
the reference): start `FROM` the stock **4.1.0** base, install `rdma-core
ibverbs-providers libibverbs-dev librdmacm-dev libucx-dev ucx-utils`, then rebuild
the **same** OpenMPI version `./configure --prefix=/usr/local/openmpi
--with-ucx=/usr --enable-mpirun-prefix-by-default`. (On a hypothetical Ubuntu
24.04 base, drop the `=/usr` and use bare `--with-ucx` so configure finds `libucp`
via pkg-config under the multiarch libdir.) To confirm IB is live at runtime,
force UCX (`OMPI_MCA_pml=ucx`) and check the job logs show `pml=ucx` with zero
`btl:tcp` lines — the [OSU example](../examples/OSU) does exactly this.
