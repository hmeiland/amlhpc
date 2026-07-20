# OSU Micro-Benchmarks — verify the InfiniBand interconnect

The [OSU Micro-Benchmarks](https://mvapich.cse.ohio-state.edu/benchmarks/) are the standard way
to measure MPI point-to-point **latency** and **bandwidth**. On Azure, the HB-series VMs
(`HB120rs_v3`, `HB176rs_v4`, ...) carry a Mellanox/NVIDIA **InfiniBand** interconnect, and these
two tests are the quickest way to confirm that MPI traffic between two VMs actually crosses IB
(low microsecond latency, ~hundreds of Gb/s bandwidth) rather than falling back to TCP/Ethernet.

The benchmarks ship pre-built in the **`openmpi-eessi`** environment
(see [environments/openmpi-eessi](../../environments/openmpi-eessi)). That image starts from the
stock AzureML `openmpi4.1.0-ubuntu22.04` base and rebuilds its OpenMPI 4.1.0 in place
`--with-ucx`, then compiles OSU 7.4 against it — so the binaries reach the InfiniBand fabric
through UCX (`pml_ucx`) while `orted` stays at `/usr/local/openmpi/bin` where AML's multi-node
launcher expects it. `osu_latency` and `osu_bw` are on `PATH`; there is nothing to mount or
`module load` for the benchmark itself.

## How AML runs the multi-node job (and why the runscripts look like this)

`amlhpc` runs a multi-node (`-N > 1`) job under an **AML MPI distribution**: AML wraps the whole
runscript in a single `mpirun` and launches it **once per rank** (one rank per node here),
handing each rank its place in one shared `MPI_COMM_WORLD`. The runscript is therefore the
**per-rank body** — it runs the OSU binary directly and must **not** call `mpirun` itself (that
would nest a second MPI job) or `parallel-ssh`.

Two consequences shape these examples:

- **Exactly one MPI program per job.** Because all ranks share one world, the first
  `MPI_Finalize` tears it down; a second MPI binary in the same script aborts in `MPI_Init` on a
  NULL communicator. Latency and bandwidth are therefore **two separate jobs / two runscripts**.
- **Force the IB transport.** Each runscript exports `OMPI_MCA_pml=ucx` (and
  `OMPI_MCA_btl=^tcp,openib`) so OpenMPI uses UCX over the mlx5 verbs device instead of silently
  falling back to the TCP btl.

## The runscripts

- **`runscript-osu-2N.sh`** — one rank on each of two VMs; runs `osu_latency` across the
  InfiniBand link. The real interconnect test.
- **`runscript-osu-bw-2N.sh`** — the companion bandwidth job (`osu_bw`), submitted separately for
  the one-MPI-program-per-job reason above.
- **`runscript-osu-1N.sh`** — both ranks inside a single VM (shared-memory transport). Does
  **not** touch IB; use it as a baseline to compare against the 2-node numbers.

## Running the InfiniBand test (2 nodes)

Submit each with `-N 2` on an HB partition, using the `openmpi-eessi` environment. `sinfo` shows
which partitions are available (the HB partition here is `hbv3`):

```
$ sbatch -p hbv3 -N 2 -e openmpi-eessi@latest ./runscript-osu-2N.sh
loyal_office_ncl4gvvhpk
$ sbatch -p hbv3 -N 2 -e openmpi-eessi@latest ./runscript-osu-bw-2N.sh
quirky_longan_rp9gxjm8kb
```

AML places one rank per node automatically for this distribution, so no hostfile or
`--map-by ppr:1:node` is needed — the two ranks already sit on different VMs. Follow the output in
the job's "Output and logs" tab (or with `sattach -f <JOBID>`).

## Results (2× Standard_HB120rs_v3, 200 Gb/s HDR InfiniBand)

Real numbers captured from the runs above. The `pml=ucx` transport was confirmed in the job logs
(zero `btl:tcp` lines), so this is genuine InfiniBand, not an Ethernet fallback.

```
=== OSU point-to-point latency (us) — VM-to-VM over InfiniBand ===
# OSU MPI Latency Test v7.4
# Size       Avg Latency(us)
1                       1.54
2                       1.53
4                       1.53
8                       1.53
16                      1.53
32                      1.69
64                      1.74
128                     1.78
256                     2.38
512                     2.44
1024                    2.59
2048                    2.78
4096                    3.34
8192                    3.94
16384                   5.11
32768                   6.78
65536                   9.33
131072                 14.09
262144                 18.07
524288                 30.07
1048576                55.09
2097152               102.72
4194304               193.81

=== OSU point-to-point bandwidth (MB/s) — VM-to-VM over InfiniBand ===
# OSU MPI Bandwidth Test v7.4
# Size      Bandwidth (MB/s)
1                       4.35
2                       8.63
4                      17.63
8                      36.04
16                     70.16
32                    140.38
64                    259.84
128                   507.04
256                   977.51
512                  1865.84
1024                 3417.05
2048                 5748.78
4096                 9048.75
8192                13387.18
16384               13265.25
32768               18130.62
65536               21549.77
131072              22640.22
262144              23252.58
524288              23423.80
1048576             23486.69
2097152             23690.80
4194304             23647.65
```

**What good looks like on HB-series IB:**

- **Latency** (small messages): ~1.5 µs one-way, as above. If you instead see tens of
  microseconds, the traffic is going over TCP/Ethernet, not IB.
- **Bandwidth** (large messages, 1–4 MB): ~23,600 MB/s, i.e. ~189 Gb/s — close to the 200 Gb/s
  HDR line rate. Single-digit-thousand MB/s again points at a non-IB fallback.

## Baseline (single node)

To see the intra-node numbers for comparison, submit the 1-node script (no `-N`):

```
$ sbatch -p hbv3 -e openmpi-eessi@latest ./runscript-osu-1N.sh
calm_river_7hd3n8vq1w
```

The 2-node latency is higher than the shared-memory baseline (the message leaves the VM), while
the 2-node bandwidth approaches the IB line rate — together confirming the interconnect between
the VMs is healthy.
