# The stock base's OpenMPI lives at /usr/local/openmpi, so AML's mpi-rsh-agent
# bridges orted across nodes; images whose mpirun/orted resolve to /usr/local/bin
# stall that executor bridge on multi-node jobs.
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential nfs-common lsb-release pssh sudo curl wget ca-certificates \
    rdma-core ibverbs-providers libibverbs-dev librdmacm-dev libucx-dev ucx-utils && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# The stock base's OpenMPI 4.1.0 is TCP-only (no UCX/verbs), so MPI traffic never
# leaves Ethernet on HB-series IB. Rebuild the SAME OpenMPI 4.1.0 in place at
# /usr/local/openmpi --with-ucx so ranks reach the InfiniBand fabric through
# pml_ucx, while orted stays at /usr/local/openmpi/bin for AML's cross-node bridge.
RUN cd /tmp && \
    wget -q https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.0.tar.gz && \
    tar xzf openmpi-4.1.0.tar.gz && cd openmpi-4.1.0 && \
    ./configure --prefix=/usr/local/openmpi --with-ucx=/usr --enable-mpirun-prefix-by-default && \
    make -j"$(nproc)" && make install && \
    ldconfig && cd / && rm -rf /tmp/openmpi-4.1.0*

## install Dask (scheduler/worker CLI) for the dask-up worker pools.
## Pinned to the login-CI azureml_py38 scheduler's distributed version so
## workers match the scheduler protocol without a job-time reinstall.
RUN pip install --no-cache-dir "dask[distributed]==2023.2.0"

## HPC tuning
RUN echo '\
*               hard    memlock         unlimited\n\
*               soft    memlock         unlimited\n\
*               hard    nofile          65535\n\
*               soft    nofile          65535\n\
*               hard    stack           unlimited\n\
*               soft    stack           unlimited' >> /etc/security/limits.conf

## install azcopy
RUN cd /usr/bin && \
        wget -q https://aka.ms/downloadazcopy-v10-linux -O - | tar zxf - --strip-components 1 --wildcards '*/azcopy' && \
        chmod 755 /usr/bin/azcopy

## install CernVM-FS + EESSI configuration (https://www.eessi.io/docs/getting_access/native_installation/)
RUN wget -q https://cvmrepo.s3.cern.ch/cvmrepo/apt/cvmfs-release-latest_all.deb && \
    dpkg -i cvmfs-release-latest_all.deb && \
    rm -f cvmfs-release-latest_all.deb && \
    apt-get update && \
    apt-get install -y cvmfs && \
    wget -q https://github.com/EESSI/filesystem-layer/releases/download/latest/cvmfs-config-eessi_latest_all.deb && \
    dpkg -i cvmfs-config-eessi_latest_all.deb && \
    rm -f cvmfs-config-eessi_latest_all.deb && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*
RUN bash -c "echo 'CVMFS_CLIENT_PROFILE="single"' > /etc/cvmfs/default.local"; \
    bash -c "echo 'CVMFS_QUOTA_LIMIT=10000' >> /etc/cvmfs/default.local"; \
    bash -c "echo 'CVMFS_HTTP_PROXY=DIRECT' >> /etc/cvmfs/default.local"; \
    cvmfs_config setup; ln -sf /bin/bash /bin/sh

RUN groupadd -g 1000 azureuser && useradd -m azureuser -u 1000 -g azureuser && adduser azureuser sudo && echo "azureuser ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/azureuser

# Do NOT add a `USER` directive: AML's cross-node executor bridge is a UDS socket
# under /mnt/azureml/cr/j/.../.grpc reachable only by root. A non-root user makes
# the leader's mpi-rsh-agent time out (30s) on remote-executor and the multi-node
# job never starts. Root also mounts cvmfs directly, so no sudo user is needed.
WORKDIR /app
