#!/bin/bash
# OSU Micro-Benchmarks - point-to-point latency between two HB-series VMs over
# the InfiniBand interconnect.
#
# Submit with two nodes on an HB partition, using the openmpi-eessi environment
# (stock AzureML OpenMPI 4.1.0 rebuilt --with-ucx + baked OSU; see
# environments/openmpi-eessi):
#   sbatch -p hbv3 -N 2 -e openmpi-eessi@latest ./runscript-osu-2N.sh
#
# amlhpc runs multi-node (-N > 1) jobs under an AML MPI distribution, so AML
# launches this script once per rank (one rank per node here) inside its own
# mpirun and hands each rank the MPI world. The script is therefore the
# per-rank body: it runs the OSU binary directly. It must NOT call mpirun itself
# (that would nest a second MPI job) or parallel-ssh.
#
# Exactly ONE MPI program may run per job: AML wraps the whole script in a single
# mpirun, so the first MPI_Finalize tears down the shared world and a second
# binary aborts in MPI_Init. Bandwidth therefore has its own runscript
# (runscript-osu-bw-2N.sh) submitted as a separate job.

# Force the InfiniBand path: pml_ucx over the mlx5 verbs device. Without this
# OpenMPI can silently fall back to the TCP btl and the numbers reflect Ethernet.
export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx
export OMPI_MCA_btl=^tcp,openib

if [ "${OMPI_COMM_WORLD_RANK}" = "0" ]; then
    echo "=== OSU point-to-point latency (us) - VM-to-VM over InfiniBand ==="
fi
osu_latency
