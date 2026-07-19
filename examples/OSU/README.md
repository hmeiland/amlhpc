# OSU Micro-Benchmarks — verify the InfiniBand interconnect

The [OSU Micro-Benchmarks](https://mvapich.cse.ohio-state.edu/benchmarks/) are the standard way
to measure MPI point-to-point **latency** and **bandwidth**. On Azure, the HB-series VMs
(`HB120rs_v2`, `HB176rs_v4`, ...) carry a Mellanox/NVIDIA **InfiniBand** interconnect, and these
two tests are the quickest way to confirm that MPI traffic between two VMs actually crosses IB
(low microsecond latency, ~hundreds of Gb/s bandwidth) rather than falling back to TCP/Ethernet.

The benchmarks come straight from the [EESSI](http://www.eessi.io/) cvmfs software stack, so
there is nothing to compile — the dependencies are already in the default container/environment
(see the [EESSI example](../EESSI/readme.md)). We use the `osu_latency` and `osu_bw`
point-to-point tests from `OSU-Micro-Benchmarks/7.2-gompi-2023b`.

## The two runscripts

- **`runscript-osu-2N.sh`** — runs one MPI rank on each of two VMs, so the traffic crosses the
  InfiniBand link. This is the real interconnect test.
- **`runscript-osu-1N.sh`** — runs both ranks inside a single VM (shared-memory transport). It
  does **not** touch IB; use it as a baseline to compare against the 2-node numbers.

## Running the InfiniBand test (2 nodes)

Submit with `-N 2` on an HB partition. `sinfo` shows which partitions are available (the HB
partition is typically named `hbv2`):

```
$ sbatch -p hbv2 -N 2 ./runscript-osu-2N.sh
brave_meadow_k2p9x7m4qz
```

The runscript mounts the EESSI stack on every node in the job (via
`parallel-ssh` over `AZ_BATCH_HOST_LIST`, the comma-separated node list AML sets for a
multi-node job), loads the OSU module, builds a hostfile with **one slot per node**, and pins
one rank to each VM so the two ranks sit on different machines:

```bash
#!/bin/bash
parallel-ssh -i -H "${AZ_BATCH_HOST_LIST//,/ }" \
    "sudo mount -t cvmfs software.eessi.io /cvmfs/software.eessi.io"

source /cvmfs/software.eessi.io/versions/2023.06/init/bash
ml load OSU-Micro-Benchmarks/7.2-gompi-2023b

rm -f hostfile.txt
for i in ${AZ_BATCH_HOST_LIST//,/ }
do
        echo "${i}:1" >> hostfile.txt
done

mpirun -x PATH -np 2 --map-by ppr:1:node --hostfile hostfile.txt osu_latency
mpirun -x PATH -np 2 --map-by ppr:1:node --hostfile hostfile.txt osu_bw
```

The `--map-by ppr:1:node` (one process-per-resource, per node) placement is what guarantees the
two ranks land on **different** VMs — without it MPI could pack both ranks onto the first node
and the numbers would reflect shared memory, not IB.

Follow the output in the job's "Output and logs" tab (or with `sattach -f <JOBID>`).

## Reading the results

`osu_latency` prints a table of message size against one-way latency in microseconds; `osu_bw`
prints message size against bandwidth in MB/s:

```
=== OSU point-to-point latency (us) — VM-to-VM over InfiniBand ===
# OSU MPI Latency Test
# Size          Latency (us)
0                        1.6x
1                        1.6x
...
=== OSU point-to-point bandwidth (MB/s) — VM-to-VM over InfiniBand ===
# OSU MPI Bandwidth Test
# Size        Bandwidth (MB/s)
1                        x.xx
...
4194304              2xxxx.xx
```

**What good looks like on HB-series IB:**

- **Latency** (small messages, e.g. 0–8 bytes): a couple of microseconds. If you instead see
  tens of microseconds, the traffic is very likely going over TCP/Ethernet, not IB.
- **Bandwidth** (large messages, 1–4 MB): tens of thousands of MB/s (HDR IB is 200 Gb/s ≈
  ~24 GB/s ≈ ~24000 MB/s). Single-digit-thousand MB/s again points at a non-IB fallback.

## Baseline (single node)

To see the intra-node numbers for comparison, submit the 1-node script (no `-N`):

```
$ sbatch -p hbv2 ./runscript-osu-1N.sh
calm_river_7hd3n8vq1w
```

The 2-node latency should be higher than this shared-memory baseline, and the 2-node bandwidth
should approach the IB line rate — confirming the interconnect between the VMs is healthy.
