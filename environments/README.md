# environments

The default amlhpc docker container is currently build and hosted on docker.io since the bicep deployment does not support
the contextUri through git yet.

```
docker build -t amlhpc-ubuntu2004 amlhpc-ubuntu2004/
docker tag amlhpc-ubuntu2004 hmeiland/amlhpc-ubuntu2004
docker push hmeiland/amlhpc-ubuntu2004
```

The amlhpc-ubuntu2004 environment references this docker container and does not use any additional conda.yml.
