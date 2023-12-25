def sbatch(vargs=None):
    import os

    try:
        subscription_id = os.environ['SUBSCRIPTION']
        resource_group = os.environ['CI_RESOURCE_GROUP']
        workspace_name = os.environ['CI_WORKSPACE']
    except Exception as error:
        print("please set the export variables: SUBSCRIPTION, CI_RESOURCE_GROUP, and CI_WORKSPACE")

    pwd = os.environ['PWD']

    from azure.ai.ml import MLClient, command, parallel, Output, Input
    from azure.identity import DefaultAzureCredential
    from azure.ai.ml.entities import Environment
    from azure.ai.ml.constants import AssetTypes, InputOutputModes
    from azure.ai.ml.sweep import Choice
    # from azure.ai.ml.parallel import parallel_run_function, RunFunction, ParallelJob
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
    parser.add_argument('--parallel', type=str, help='command line to be executed, should be enclosed with quotes')
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
    
    if (args.parallel == "parallel"):
        # parallel job to process file data
        parallel_job = parallel.parallel_run_function(
            name="test par job",
            #display_name="Batch Score with File Dataset",
            #description="parallel component for batch score",
            #inputs=dict(
            #    job_data_path=Input(
            #        type=AssetTypes.MLTABLE,
            #        description="The data to be split and scored in parallel",
            #        )
            #    ),
            #outputs=dict(job_output_path=Output(type=AssetTypes.MLTABLE)),
            #input_data="${{inputs.job_data_path}}",
            #instance_count=2,
            #mini_batch_size="1",
            #mini_batch_error_threshold=1,
            #max_concurrency_per_instance=10,
            #code="runscript.sh",
            task=parallel.RunFunction(
                code="runscript.sh",
            #    entry_script="file_batch_inference.py",
            #    program_arguments="--job_output_path ${{outputs.job_output_path}}",
            #    environment="azureml:AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:1",
            ),
        )

        returned_job = ml_client.jobs.create_or_update(parallel_job)
        print(returned_job.name)

    if (args.parallel == "sweep"):
        command_job = command(
            code=job_code,
            command=job_command,
            environment=args.environment,
            instance_count=args.nodes,
            compute=args.partition,
            outputs=outputs,
            inputs={"SLURM_ARRAY_TASK_ID": 0},
            environment_variables=job_env,
            )

        command_job_for_sweep = command_job(
            SLURM_ARRAY_TASK_ID=Choice(values=task_index_list),
            )

        sweep_job = command_job_for_sweep.sweep(
            compute=args.partition,
            sampling_algorithm="random",
            primary_metric="None",
            goal="Minimize",
            )

        sweep_job.set_limits(max_concurrent_trials=2)
        #sweep_job.settings(max_concurrency_per_instance=2)

        returned_job = ml_client.jobs.create_or_update(sweep_job)
        print(returned_job.name)

    else:
        #for index in range(array_list[0], array_list[1], array_list[2]):
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

        #command_job.set_resources(
            #instance_type="STANDARD_D2_v2",
            #properties={"key": "new_val"},
            #shm_size="3g",
        #    )
        #command_job.set_limits(
        #    timeout=600,
        #    max_nodes=4,
        #    )

        returned_job = ml_client.jobs.create_or_update(command_job)
        print(returned_job.name)
