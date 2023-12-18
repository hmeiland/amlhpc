def sinfo(vargs):
    import os
    from azureml.core import Workspace
    from azureml.core.authentication import AzureCliAuthentication
    from azureml.core.compute import ComputeTarget, AmlCompute

    try:
        subscription_id = os.environ['SUBSCRIPTION']
        resource_group = os.environ['CI_RESOURCE_GROUP']
        workspace_name = os.environ['CI_WORKSPACE']
    except Exception as error:
        print("please set the export variables: SUBSCRIPTION, CI_RESOURCE_GROUP, and CI_WORKSPACE")

    cli_auth = AzureCliAuthentication()
    ws = Workspace(
        subscription_id=subscription_id,
        resource_group=resource_group,
        workspace_name=workspace_name,
        auth=cli_auth
        )

    sinfo_list = ComputeTarget.list(workspace=ws)
    print("PARTITION\tAVAIL\tVM_SIZE\t\t\tNODES\tSTATE")
    for i in sinfo_list:
        line = (AmlCompute.get(i).get('name'))
        if len(line) < 8:
            line += "\t"
        line += "\tUP\t" + AmlCompute.get_status(i).vm_size + "\t"
        if len(line.expandtabs()) < 41:
            line += "\t"
        line += str(AmlCompute.get(i).get('properties', {}).get('properties', {}).get('scaleSettings', {}).get('maxNodeCount'))
        print(line)


def squeue(vargs):
    import os

    try:
        subscription_id = os.environ['SUBSCRIPTION']
        resource_group = os.environ['CI_RESOURCE_GROUP']
        workspace_name = os.environ['CI_WORKSPACE']
    except Exception as error:
        print("please set the export variables: SUBSCRIPTION, CI_RESOURCE_GROUP, and CI_WORKSPACE")

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
        if len(line) < 24:
            line += "\t"
        line += "\t" + job.display_name
        if len(line) < 8:
            line += "\t"
        line += "\t" + str(job.compute)
        print(line)


def sbatch(vargs):
    import os

    try:
        subscription_id = os.environ['SUBSCRIPTION']
        resource_group = os.environ['CI_RESOURCE_GROUP']
        workspace_name = os.environ['CI_WORKSPACE']
    except Exception as error:
        print("please set the export variables: SUBSCRIPTION, CI_RESOURCE_GROUP, and CI_WORKSPACE")

    pwd = os.environ['PWD']

    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
        )

    from azure.ai.ml import command
    import argparse

    parser = argparse.ArgumentParser(description='sbatch: submit jobs to Azure Machine Learning')
    parser.add_argument('-a', '--array', default="None", type=str, help='index for array jobs')
    parser.add_argument('-p', '--partition', type=str, required=True,
                        help='set compute partition where the job should be run. Use <sinfo> to view available partitions')
    parser.add_argument('-N', '--nodes', default=1, type=int, help='amount of nodes to use for the job')
    parser.add_argument('-w', '--wrap', type=str, help='command line to be executed, should be enclosed with quotes')
    parser.add_argument('script', nargs='?', default="None", type=str, help='script to be executed')
    args = parser.parse_args(vargs)

    if (args.script == "None") and (args.wrap is None):
        print("Missing: provide either script to execute as argument or commandline to execute through --wrap option")
        exit(-1)

    if (args.script != "None") and (args.wrap is not None):
        print("Conflict: provide either script to execute as argument or commandline to execute through --wrap option")
        exit(-1)

    if (args.script != "None"):
        command_job = command(
            code=pwd + "/" + args.script,
            command=args.script,
            environment="docker-test1:4",
            instance_count=args.nodes,
            compute=args.partition,
            )

    if (args.wrap is not None):
        command_job = command(
            command=args.wrap,
            environment="docker-test1:4",
            instance_count=args.nodes,
            compute=args.partition,
            )

    returned_job = ml_client.jobs.create_or_update(command_job)
    print(returned_job.name)
