FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    nfs-common lsb-release && \
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
RUN wget https://github.com/EESSI/filesystem-layer/releases/download/v0.4.0/cvmfs-config-eessi_0.4.0_all.deb && dpkg -i cvmfs-config-eessi_0.4.0_all.deb
RUN bash -c "echo 'CVMFS_CLIENT_PROFILE="single"' > /etc/cvmfs/default.local"; \
    bash -c "echo 'CVMFS_QUOTA_LIMIT=10000' >> /etc/cvmfs/default.local"; \
    bash -c "echo 'CVMFS_REPOSITORIES=cms.cern.ch,pilot.eessi-hpc.org' >> /etc/cvmfs/default.local"; \
    bash -c "echo 'CVMFS_HTTP_PROXY=DIRECT' >> /etc/cvmfs/default.local"; \
    cvmfs_config setup; mkdir -p /cvmfs/pilot.eessi-hpc.org; ln -sf /bin/bash /bin/sh
WORKDIR /app
