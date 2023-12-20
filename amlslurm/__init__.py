def sinfo(vargs=None):
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

    sinfo_list = ml_client.compute.list(compute_type="AMLCompute")

    print("PARTITION\tAVAIL\tVM_SIZE\t\t\tNODES\tSTATE")
    for i in sinfo_list:
        line = i.name
        if len(line) < 8:
            line += "\t"
        line += "\tUP\t" + i.size + "\t"
        if len(line.expandtabs()) < 41:
            line += "\t"
        line += str(i.max_instances)
        print(line)


def squeue(vargs=None):
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


def sbatch(vargs=None):
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
    from azure.ai.ml.entities import Environment
    import argparse

    parser = argparse.ArgumentParser(description='sbatch: submit jobs to Azure Machine Learning')
    parser.prog = "sbatch"
    parser.add_argument('-a', '--array', default="None", type=str, help='index for array jobs')
    parser.add_argument('--container', default="None", type=str, help='container environment for the job to run in')
    parser.add_argument('--datamover', default="None", type=str, help='use "simple" for moving the (recursive) data along with the runscript')
    parser.add_argument('-e', '--environment', default="None", type=str, help='Azure Machine Learning environment, should be enclosed in quotes, may use @latest')
    parser.add_argument('-N', '--nodes', default=1, type=int, help='amount of nodes to use for the job')
    parser.add_argument('-p', '--partition', type=str, required=True,
                        help='set compute partition where the job should be run. Use <sinfo> to view available partitions')
    parser.add_argument('-w', '--wrap', type=str, help='command line to be executed, should be enclosed with quotes')
    parser.add_argument('script', nargs='?', default="None", type=str, help='runscript to be executed')
    args = parser.parse_args(vargs)

    if (args.script == "None") and (args.wrap is None):
        print("Missing: provide either script to execute as argument or commandline to execute through --wrap option")
        exit(-1)

    if (args.script != "None") and (args.wrap is not None):
        print("Conflict: provide either script to execute as argument or commandline to execute through --wrap option")
        exit(-1)

    if (args.container != "None"):
        env_docker_image = Environment(
                image=args.container,
                name="sbatch-container-image",
                description="Environment created from a Docker image.",
                )
        ml_client.environments.create_or_update(env_docker_image)
        args.environment = "sbatch-container-image@latest"

    if (args.environment == "None"):
        args.environment = "ubuntu2004-mofed@latest"

    if (args.script != "None"):
        job_code = pwd + "/" + args.script
        job_command = args.script

    if (args.wrap is not None):
        job_code = None 
        job_command = args.wrap

    if (args.datamover == "simple"):
        job_code = pwd + "/"

    command_job = command(
        code=job_code,
        command=job_command,
        environment=args.environment,
        instance_count=args.nodes,
        compute=args.partition,
        )

    returned_job = ml_client.jobs.create_or_update(command_job)
    print(returned_job.name)
