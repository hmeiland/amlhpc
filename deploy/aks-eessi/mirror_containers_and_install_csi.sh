#!/bin/bash

echo "usage: mirror_containers.sh <ACR_name>"

az acr login -n $1

docker pull registry.cern.ch/kubernetes/cvmfs-csi:v2.3.2
docker tag registry.cern.ch/kubernetes/cvmfs-csi:v2.3.2 ${1}.azurecr.io/kubernetes/cvmfs-csi:v2.3.2
docker push ${1}.azurecr.io/kubernetes/cvmfs-csi:v2.3.2

docker pull registry.k8s.io/sig-storage/csi-node-driver-registrar:v2.8.0
docker tag registry.k8s.io/sig-storage/csi-node-driver-registrar:v2.8.0 ${1}.azurecr.io/sig-storage/csi-node-driver-registrar:v2.8.0
docker push ${1}.azurecr.io/sig-storage/csi-node-driver-registrar:v2.8.0

docker pull registry.k8s.io/sig-storage/csi-provisioner:v3.5.0
docker tag registry.k8s.io/sig-storage/csi-provisioner:v3.5.0 ${1}.azurecr.io/sig-storage/csi-provisioner:v3.5.0
docker push ${1}.azurecr.io/sig-storage/csi-provisioner:v3.5.0


git clone -b release-2.4 https://github.com/cvmfs-contrib/cvmfs-csi.git
pushd cvmfs-csi/deployments/helm/cvmfs-csi
sed -i "s#registry.cern.ch#${1}.azurecr.io#g" values.yaml
sed -i "s#registry.k8s.io#${1}.azurecr.io#g" values.yaml
sed -i 's#CVMFS_HTTP_PROXY=.*$#CVMFS_HTTP_PROXY=DIRECT#g' values.yaml
helm package .
helm push cvmfs-csi-2.1.1.tgz oci://${1}.azurecr.io/helm
popd

helm install cvmfs oci://${1}.azurecr.io/helm/cvmfs-csi
kubectl create -f aml-cvmfs-pvc.yaml

rm -rf cvmfs-csi
