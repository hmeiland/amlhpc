FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    nfs-common lsb-release pssh sudo && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*  

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

## install eessi
RUN wget --no-check-certificate https://ecsft.cern.ch/dist/cvmfs/cvmfs-release/cvmfs-release-latest_all.deb && dpkg -i cvmfs-release-latest_all.deb
RUN apt-get update && \
    apt-get install -y cvmfs && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*  
RUN wget https://github.com/EESSI/filesystem-layer/releases/download/v0.5.0/cvmfs-config-eessi_0.5.0_all.deb && dpkg -i cvmfs-config-eessi_0.5.0_all.deb
RUN bash -c "echo 'CVMFS_CLIENT_PROFILE="single"' > /etc/cvmfs/default.local"; \
    bash -c "echo 'CVMFS_QUOTA_LIMIT=10000' >> /etc/cvmfs/default.local"; \
    bash -c "echo 'CVMFS_REPOSITORIES=cms.cern.ch,pilot.eessi-hpc.org,software.eessi.io' >> /etc/cvmfs/default.local"; \
    bash -c "echo 'CVMFS_HTTP_PROXY=DIRECT' >> /etc/cvmfs/default.local"; \
    cvmfs_config setup; mkdir -p /cvmfs/pilot.eessi-hpc.org /cvmfs/software.eessi.io; ln -sf /bin/bash /bin/sh

RUN groupadd -g 1000 azureuser && useradd -m azureuser -u 1000 -g azureuser && adduser azureuser sudo && echo "azureuser ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/azureuser
USER azureuser
WORKDIR /app
