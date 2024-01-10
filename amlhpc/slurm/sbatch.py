class mlComputeAuth:
    def get_token(scopes="my_scope", claims="my_claim", tenant_id="my_tenant"):
        import requests
        import os
        from azure.core.credentials import AccessToken
        resource = "https://management.azure.com"
        client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID", None)
        resp = requests.get(f"{os.environ['MSI_ENDPOINT']}?resource={resource}&clientid={client_id}&api-version=2017-09-01", headers={'Secret': os.environ["MSI_SECRET"]})
        resp.raise_for_status()
        my_token = AccessToken(resp.json()["access_token"], int(resp.json()["expires_on"]))

        return my_token


def sbatch(vargs=None):
    import os

    try:
        subscription_id = os.environ['SUBSCRIPTION']
        resource_group = os.environ['CI_RESOURCE_GROUP']
        workspace_name = os.environ['CI_WORKSPACE']
    except Exception as error:
        print("please set the export variables: SUBSCRIPTION, CI_RESOURCE_GROUP and CI_WORKSPACE")
        exit(-1)

    pwd = os.environ['PWD']

    from azure.ai.ml import MLClient, Output, Input, command
    from azure.identity import DefaultAzureCredential
    from azure.ai.ml.entities import Environment, CommandJob
    from azure.ai.ml.constants import AssetTypes, InputOutputModes
    from azure.ai.ml.sweep import SweepJob, SweepJobLimits, Choice, Objective
    import argparse
    import re

    # disable output for experimental classes
    import logging
    logging.getLogger('azure.ai.ml._utils').setLevel(logging.CRITICAL)

    try:
        on_aml = os.environ['APPSETTING_WEBSITE_SITE_NAME']
        if (on_aml == 'AMLComputeInstance'):
            credential = mlComputeAuth()
    except:
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
    #parser.add_argument('--parallel', default="single", type=str, help='command line to be executed, should be enclosed with quotes')
    parser.add_argument('-v', '--verbose', action='count', default=0,  help='provide output on found settings and job properties')
    parser.add_argument('-w', '--wrap', type=str, help='command line to be executed, should be enclosed with quotes')
    parser.add_argument('script', nargs='?', default="None", type=str, help='runscript to be executed')
    args = parser.parse_args(vargs)
    args.parallel="single"

    job_env = { "SLURM_JOB_NODES": args.nodes }

    if (args.script == "None") and (args.wrap is None):
        print("Missing: provide either script to execute as argument or commandline to execute through --wrap option")
        exit(-1)

    if (args.script != "None") and (args.wrap is not None):
        print("Conflict: provide either script to execute as argument or commandline to execute through --wrap option")
        exit(-1)

    if (args.container != "None") and (args.environment != "None"):
        print("Conflict: provide either container or environment, cannot be used together")
        exit(-1)

    if (args.container != "None"):
        if (args.verbose): print("using container: " + args.container) 
        env_docker_image = Environment(
                image=args.container,
                name="sbatch-container-image",
                description="Environment created from a Docker image.",
                )
        ml_client.environments.create_or_update(env_docker_image)
        args.environment = "sbatch-container-image@latest"
        if (args.verbose): print("created container environment: " + args.environment) 

    if (args.environment == "None"):
        args.environment = "amlhpc-ubuntu2004@latest"
        if (args.verbose): print("using default environment: " + args.environment) 

    if (args.verbose): print("using environment: " + args.environment) 

    if (args.script != "None"):
        job_code = pwd + "/" + args.script
        job_command = args.script
        if (args.verbose): 
            print("provided script to be executed: " + job_code) 
            print("provided script to be uploaded: " + job_code) 
            print("initial command (through runscript): " + job_command) 

    if (args.wrap is not None):
        job_code = None 
        job_command = args.wrap
        if (args.verbose): 
            print("no script to be uploaded.") 
            print("initial command (through wrapped script): " + job_command) 

    if (args.datamover == "simple"):
        job_code = pwd + "/"
        if (args.verbose): 
            print("selected datamover: " + args.datamover) 
            print("provided directory to be uploaded: " + job_code) 
    
    outputs = {}
    if (args.datamover == "datastore"):
        if (args.verbose): 
            print("selected datamover: " + args.datamover) 
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
        if (args.verbose): 
            print("selected datamover: " + args.datamover) 
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

    if (args.array != "None"):
        if (args.verbose): 
            print("selected array job with: " + args.array) 
        array_list_1=re.split('%', args.array)
        if len(array_list_1) == 2:
            max_nodes=int(array_list_1[1])
            if (args.verbose): 
                print("selected max nodes for array job: " + max_nodes) 
        array_list_2=re.split(':', array_list_1[0])
        if len(array_list_2) == 2:
            step=int(array_list_2[1])
        else:
            step=1
        if (args.verbose): 
            print("selected step for array range: " + step) 
        array_list_3=re.split('-', array_list_2[0])
        array_list_4=re.split(',', array_list_3[0])
        task_index_list = [eval(i) for i in array_list_4]
        if len(array_list_3) == 2:
            array_end=int(array_list_3[1])
            #print(array_end)
            array_start=int(task_index_list.pop())
            #print(array_start)
            #print(str(array_start) + " to " + str(array_end) + " step " + str(step))
            array_end += 1
            for index in range(array_start, array_end, step):
                task_index_list.append(index)
            job_env["SLURM_ARRAY_TASK_STEP"] = step
        task_index_list.sort()
        job_env["SLURM_ARRAY_TASK_COUNT"] = len(task_index_list)
        job_env["SLURM_ARRAY_TASK_MAX"] = task_index_list[-1]
        job_env["SLURM_ARRAY_TASK_MIN"] = task_index_list[0]
        #print(task_index_list)
        #print(len(task_index_list))
        if len(array_list_1) == 1:
            max_nodes=len(task_index_list)
        job_env["SLURM_JOB_NUM_NODES"] = max_nodes
        args.parallel="sweepjob"
    
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

        sweep_job.set_limits(max_concurrent_trials=max_nodes)

        returned_job = ml_client.jobs.create_or_update(sweep_job)
        print(returned_job.name)

    if (args.parallel == "sweepjob"):
        command_job = CommandJob(
            compute=args.partition,
            environment=args.environment,
            code=job_code,
            command="export SLURM_ARRAY_TASK_ID=`echo $AZUREML_SWEEP_SLURM_ARRAY_TASK_ID`; " +job_command,
            environment_variables=job_env,
            )

        sweep_job = SweepJob(
            sampling_algorithm="random",
            trial=command_job,
            search_space={ "SLURM_ARRAY_TASK_ID": Choice(values=task_index_list)},
            compute=args.partition,
            limits=SweepJobLimits(max_concurrent_trials=max_nodes),
            objective=Objective(goal="maximize", primary_metric="none"),
        )

        returned_job = ml_client.jobs.create_or_update(sweep_job)
        print(returned_job.name)

    if (args.parallel == "single"):
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
