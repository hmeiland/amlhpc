#!/bin/bash
# OSU Micro-Benchmarks — single-node baseline (both MPI ranks on ONE VM).
#
# Submit on a single node with the openmpi-eessi environment:
#   sbatch -p hbv3 -e openmpi-eessi@latest ./runscript-osu-1N.sh
#
# This runs entirely inside one VM (shared-memory / intra-node transport), so
# it does NOT exercise InfiniBand. Use it as a baseline: the 2-node run should
# show higher latency and (near line-rate) IB bandwidth by comparison.
#
# A single-node job is NOT launched under AML's MPI distribution (see
# amlhpc/slurm/sbatch.py), so unlike the 2-node runscripts this script drives
# mpirun itself to place the two ranks; osu_latency and osu_bw each get their
# own mpirun and therefore their own MPI world.

echo "=== OSU point-to-point latency (us) — intra-node baseline ==="
mpirun -np 2 osu_latency

echo "=== OSU point-to-point bandwidth (MB/s) — intra-node baseline ==="
mpirun -np 2 osu_bw
