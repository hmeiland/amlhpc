FROM ubuntu:22.04 as builder

COPY tNavigator-setup-Linux-64.sh .
RUN export TERM=linux; chmod +x tNavigator-setup-Linux-64.sh; ./tNavigator-setup-Linux-64.sh --noexec

FROM ubuntu:22.04 

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    locales && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* && \
    locale-gen en_US.utf8

RUN mkdir -p /opt/tNavigator-Linux-64

COPY --from=builder ./tNavigator-Linux-64/tNavigator-con /opt/tNavigator-Linux-64/
COPY --from=builder ./tNavigator-Linux-64/tNavigator-con-mpi /opt/tNavigator-Linux-64/

