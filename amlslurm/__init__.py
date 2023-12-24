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

    sinfo_list = ml_client.compute.list()
    #sinfo_list = ml_client.compute.list(compute_type="AMLCompute")
    #sinfo_usage_list = ml_client.compute.list_usage()

    print("PARTITION\tAVAIL\tVM_SIZE\t\t\tNODES\tSTATE")
    for i in sinfo_list:
        #print(i)
        line = i.name
        if len(line) < 8:
            line += "\t"
        try:
            line += "\tUP\t" + i.size + "\t"
        except:
            line += "\tUP\t" + "unknown" + "\t\t"
        if len(line.expandtabs()) < 41:
            line += "\t"
        try:
            line += str(i.max_instances) + "\t"
        except:
            line += "unknown" + "\t"
        try:
            line += str(i.state)
        except:
            line += "unknown"
        print(line)

    #for i in sinfo_usage_list:
    #    print(i.type + " " + str(i.current_value))


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

    from azure.ai.ml import MLClient, command, Output
    from azure.identity import DefaultAzureCredential
    from azure.ai.ml.entities import Environment
    from azure.ai.ml.constants import AssetTypes, InputOutputModes
    import argparse
    import re

    import logging
    logging.getLogger('azure.ai.ml._utils').setLevel(logging.CRITICAL)


    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
        enable_telemetry=False,
        )

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

    job_env = { "SLURM_JOB_NODES": args.nodes }
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
        args.environment = "amlslurm-ubuntu2004@latest"

    if (args.script != "None"):
        job_code = pwd + "/" + args.script
        job_command = args.script

    if (args.wrap is not None):
        job_code = None 
        job_command = args.wrap

    if (args.datamover == "simple"):
        job_code = pwd + "/"
    
    outputs = {}
    if (args.datamover == "datastore"):
        # print(pwd)
        data_stores_list = ml_client.datastores.list()
        data_stores_name_list = []
        for j in data_stores_list:
            data_stores_name_list.append(j.name)
        pwd_list = pwd.split("/")
        datastore=(list(set(data_stores_name_list).intersection(pwd_list)))
        if not datastore:
            print("Can not find a likely datastore, is it e.g. mounted as a different name?")
            exit(-1)
        # print("datastore found: " + datastore[0])
        datastore_index = pwd_list.index(datastore[0])
        datastore_pwd = ('/'.join(pwd_list[datastore_index+1:]))
        # print("relative pwd: " + datastore_pwd)
        output_path = "azureml://datastores/" + datastore[0] + "/paths/" + datastore_pwd
        outputs = {"job_workdir": Output(type=AssetTypes.URI_FOLDER, path=output_path, mode=InputOutputModes.RW_MOUNT)}
        job_code = None
        job_command = "cd $AZURE_ML_OUTPUT_JOB_WORKDIR; " + job_command

    if (args.datamover == "nfs"):
        # print(pwd)
        pwd_list = pwd.split("/")
        while pwd_list:
            if (os.path.ismount('/'.join(pwd_list))): break
            pwd_list.pop()
        # print('/'.join(pwd_list))
        os_stream = os.popen('mount -t nfs')
        os_output = os_stream.read()
        for line in os_output.splitlines():
            # print(line)
            words = line.split(" ")
            # print(words[2])
            if ('/'.join(pwd_list) == words[2]): break
        # print("mount -t nfs " + words[0] + " " + words[2])
        if (args.nodes > 1):
            start_command = "parallel-ssh -i -H \"${AZ_BATCH_HOST_LIST//,/ }\" "
        start_command += "\"mkdir -p " + words[2] + "; mount -t nfs " + words[0] + " " + words[2] + "\" ; cd " + pwd + "; "
        job_command = start_command + job_command
        job_code = None

    array_list = [0, 1, 1]
    if (args.array != "None"):
        array_list=re.split('-|:', args.array)
        if len(array_list) == 2:
            array_list.append('1')
        array_list = [eval(i) for i in array_list]
        print(str(array_list[0]) + " to " + str(array_list[1]) + " step " + str(array_list[2]))
        array_list[1] += 1
        task_index_list = []
        for index in range(array_list[0], array_list[1], array_list[2]):
            task_index_list.append(index)
        print(task_index_list)
        print(len(task_index_list))
        job_env["SLURM_ARRAY_TASK_COUNT"] = len(task_index_list)
    
    for index in range(array_list[0], array_list[1], array_list[2]):
        job_env["SLURM_ARRAY_TASK_ID"] = index

        command_job = command(
            code=job_code,
            command=job_command,
            environment=args.environment,
            instance_count=args.nodes,
            compute=args.partition,
            outputs=outputs,
            environment_variables=job_env,
            )

        command_job.set_resources(
            #instance_type="STANDARD_D2_v2",
            #properties={"key": "new_val"},
            #shm_size="3g",
            )

        returned_job = ml_client.jobs.create_or_update(command_job)
        print(returned_job.name)
