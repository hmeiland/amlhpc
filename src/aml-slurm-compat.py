#!/usr/bin/env python3

import os,sys
from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.compute import ComputeTarget, AmlCompute



cli_auth = AzureCliAuthentication()
ws = Workspace(
    #subscription_id="f5a67d06-2d09-4090-91cc-e3298907a021",
    #resource_group="hugo-ml",
    #workspace_name="hugo-eessi",
    subscription_id=os.environ['SUBSCRIPTION'],
    resource_group=os.environ['RESOURCEGROUP'],
    workspace_name=os.environ['WORKSPACE'],
    auth=cli_auth
    )

def sinfo():
    sinfo_list = ComputeTarget.list(workspace=ws)
    print("PARTITION\tAVAIL\tVM_SIZE\t\t\tNODES\tSTATE")
    for i in sinfo_list:
        line = (AmlCompute.get(i).get('name'))
        if len(line) < 8: line += "\t"
        line += "\tUP\t" + AmlCompute.get_status(i).vm_size + "\t"
        if len(line.expandtabs()) < 41 : line += "\t"
        line += str(AmlCompute.get(i).get('properties',{}).get('properties',{}).get('scaleSettings',{}).get('maxNodeCount'))
        print(line)

if sys.argv[0].rsplit('/', 1)[-1] == "sinfo":
    sinfo()
    exit()


