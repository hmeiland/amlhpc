#!/bin/bash
# OSU Micro-Benchmarks — point-to-point latency and bandwidth between two
# HB-series VMs over the InfiniBand interconnect.
#
# Submit with two nodes:
#   sbatch -p hbv2 -N 2 ./runscript-osu-2N.sh
#
# AZ_BATCH_HOST_LIST is a comma-separated list of the node IPs AML assigns to
# the job. We mount the EESSI cvmfs stack on every node, then run one MPI rank
# per node so the two ranks land on different VMs and the traffic crosses IB.

# Mount the EESSI software stack on all nodes in the job.
parallel-ssh -i -H "${AZ_BATCH_HOST_LIST//,/ }" \
    "sudo mount -t cvmfs software.eessi.io /cvmfs/software.eessi.io"

source /cvmfs/software.eessi.io/versions/2023.06/init/bash
ml load OSU-Micro-Benchmarks/7.2-gompi-2023b

# One line per node so mpirun can place exactly one rank on each VM.
rm -f hostfile.txt
for i in ${AZ_BATCH_HOST_LIST//,/ }
do
        echo "${i}:1" >> hostfile.txt
done

echo "=== hosts ==="
cat hostfile.txt

echo "=== OSU point-to-point latency (us) — VM-to-VM over InfiniBand ==="
mpirun -x PATH -np 2 --map-by ppr:1:node --hostfile hostfile.txt osu_latency

echo "=== OSU point-to-point bandwidth (MB/s) — VM-to-VM over InfiniBand ==="
mpirun -x PATH -np 2 --map-by ppr:1:node --hostfile hostfile.txt osu_bw
