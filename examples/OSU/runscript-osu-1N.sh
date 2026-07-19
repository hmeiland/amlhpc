#!/bin/bash
# OSU Micro-Benchmarks — single-node baseline (both MPI ranks on ONE VM).
#
# Submit on a single node:
#   sbatch -p hbv2 ./runscript-osu-1N.sh
#
# This runs entirely inside one VM (shared-memory / intra-node transport), so
# it does NOT exercise InfiniBand. Use it as a baseline: the 2-node run should
# show higher latency and (near line-rate) IB bandwidth by comparison.

sudo mount -t cvmfs software.eessi.io /cvmfs/software.eessi.io
source /cvmfs/software.eessi.io/versions/2023.06/init/bash
ml load OSU-Micro-Benchmarks/7.2-gompi-2023b

echo "=== OSU point-to-point latency (us) — intra-node baseline ==="
mpirun -np 2 osu_latency

echo "=== OSU point-to-point bandwidth (MB/s) — intra-node baseline ==="
mpirun -np 2 osu_bw
