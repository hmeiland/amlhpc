#!/bin/bash
# OSU Micro-Benchmarks - point-to-point bandwidth between two HB-series VMs over
# the InfiniBand interconnect. Companion to runscript-osu-2N.sh (latency).
#
# Submit as a SEPARATE two-node job (see runscript-osu-2N.sh for why latency and
# bandwidth cannot share one job under AML's MPI distribution):
#   sbatch -p hbv3 -N 2 -e openmpi-eessi@latest ./runscript-osu-bw-2N.sh

# Force the InfiniBand path: pml_ucx over the mlx5 verbs device. Without this
# OpenMPI can silently fall back to the TCP btl and the numbers reflect Ethernet.
export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx
export OMPI_MCA_btl=^tcp,openib

if [ "${OMPI_COMM_WORLD_RANK}" = "0" ]; then
    echo "=== OSU point-to-point bandwidth (MB/s) - VM-to-VM over InfiniBand ==="
fi
osu_bw
