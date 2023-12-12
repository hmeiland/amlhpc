#!/usr/bin/env python3

import os,sys
from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.compute import ComputeTarget, AmlCompute

subscription_id=os.environ['SUBSCRIPTION']
resource_group=os.environ['CI_RESOURCE_GROUP']
workspace_name=os.environ['CI_WORKSPACE']


cli_auth = AzureCliAuthentication()
ws = Workspace(
    subscription_id=os.environ['SUBSCRIPTION'],
    resource_group=os.environ['CI_RESOURCE_GROUP'],
    workspace_name=os.environ['CI_WORKSPACE'],
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


def squeue():
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
        )
    job_list = []
    list_of_jobs = ml_client.jobs.list()
    for job in list_of_jobs:
        job_list.append(job)
    print("JOBID\t\t\t\tNAME\t\tPARTITION\tSTATE\tTIME")
    for job in job_list:
        line = job.name
        if len(line) < 24: line += "\t"
        line += "\t" + job.display_name
        if len(line) < 8: line += "\t"
        line += "\t" + str(job.compute)
        #line += "\t" + str(job.properties.EndTimeUtc)
        #line += "\t" + str(job.creation_context.created_by)
        print(line)

def sbatch():
    from azure.ai.ml import command
    command_job = command(
        code="/home/azureuser/cloudfiles/code/Users/gaobrien/GeophysicsAML/gsobrien_F_shot_generate.py",
        command="hostname",
        environment= "docker-test1:4",
        instance_count=1,
        compute="hc44",
        display_name="testjob"
        )
    returned_job = ml_client.jobs.create_or_update(command_job)
    print(returned_job.name)



if sys.argv[0].rsplit('/', 1)[-1] == "sinfo":
    sinfo()
    exit()
if sys.argv[0].rsplit('/', 1)[-1] == "squeue":
    squeue()
    exit()
if sys.argv[0].rsplit('/', 1)[-1] == "sbatch":
    sbatch()
    exit()


