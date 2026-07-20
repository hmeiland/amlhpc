# Build on the stock AzureML openmpi4.1.0-ubuntu22.04 base. Its OpenMPI lives at
# /usr/local/openmpi, so AML's mpi-rsh-agent bridges orted across nodes on hbv3;
# images whose mpirun/orted resolve to /usr/local/bin stall that executor bridge.
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential nfs-common lsb-release pssh sudo curl wget ca-certificates \
    rdma-core ibverbs-providers libibverbs-dev librdmacm-dev libucx-dev ucx-utils && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# The stock base's OpenMPI 4.1.0 is TCP-only (no UCX/verbs), so its OSU numbers
# never leave Ethernet. Rebuild the SAME OpenMPI 4.1.0 in place at
# /usr/local/openmpi --with-ucx so ranks reach the HB-series InfiniBand fabric
# through pml_ucx, while orted stays at /usr/local/openmpi/bin for AML's bridge.
# Keeping one 4.1.0 runtime avoids mixing AML's 4.1.0 orted with a foreign libmpi.
RUN cd /tmp && \
    wget -q https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.0.tar.gz && \
    tar xzf openmpi-4.1.0.tar.gz && cd openmpi-4.1.0 && \
    ./configure --prefix=/usr/local/openmpi --with-ucx=/usr --enable-mpirun-prefix-by-default && \
    make -j"$(nproc)" && make install && \
    ldconfig && cd / && rm -rf /tmp/openmpi-4.1.0*

# CernVM-FS + EESSI configuration (https://www.eessi.io/docs/getting_access/native_installation/)
RUN wget -q https://cvmrepo.s3.cern.ch/cvmrepo/apt/cvmfs-release-latest_all.deb && \
    dpkg -i cvmfs-release-latest_all.deb && \
    rm -f cvmfs-release-latest_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends cvmfs && \
    wget -q https://github.com/EESSI/filesystem-layer/releases/download/latest/cvmfs-config-eessi_latest_all.deb && \
    dpkg -i cvmfs-config-eessi_latest_all.deb && \
    rm -f cvmfs-config-eessi_latest_all.deb && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*
RUN bash -c "echo 'CVMFS_CLIENT_PROFILE=\"single\"' > /etc/cvmfs/default.local"; \
    bash -c "echo 'CVMFS_QUOTA_LIMIT=10000' >> /etc/cvmfs/default.local"; \
    bash -c "echo 'CVMFS_HTTP_PROXY=DIRECT' >> /etc/cvmfs/default.local"; \
    cvmfs_config setup; ln -sf /bin/bash /bin/sh

# EESSI auto-mounts under /cvmfs/<repo>; an explicit `mount -t cvmfs` needs the
# per-repo mountpoint to already exist, so create it at build time.
RUN mkdir -p /cvmfs/software.eessi.io

# OSU 7.4 built against the rebuilt UCX-enabled /usr/local/openmpi so the binaries
# are ABI-matched to the OpenMPI AML's launcher places the ranks in AND can use IB.
RUN cd /tmp && \
    wget -q https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.4.tar.gz && \
    tar xzf osu-micro-benchmarks-7.4.tar.gz && \
    cd osu-micro-benchmarks-7.4 && \
    ./configure CC=/usr/local/openmpi/bin/mpicc CXX=/usr/local/openmpi/bin/mpicxx --prefix=/opt/osu && \
    make -j"$(nproc)" && make install && \
    cd / && rm -rf /tmp/osu-micro-benchmarks-7.4*
ENV PATH="/opt/osu/libexec/osu-micro-benchmarks/mpi/pt2pt:/opt/osu/libexec/osu-micro-benchmarks/mpi/collective:${PATH}"

# Do NOT add a `USER` directive: AML's cross-node executor bridge is a UDS socket
# under /mnt/azureml/cr/j/.../.grpc reachable only by root. A non-root user makes
# the leader's mpi-rsh-agent time out (30s) on remote-executor and the multi-node
# job never starts. Root also mounts cvmfs directly, so no sudo user is needed.
WORKDIR /app
