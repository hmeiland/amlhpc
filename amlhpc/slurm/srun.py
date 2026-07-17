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


def srun(vargs=None):
    """Run a job on the login ComputeInstance instead of an AmlCompute cluster.

    Mirrors sbatch, but the compute target is a ComputeInstance. With no -p the login
    CI is auto-discovered (the single computeinstance in the workspace). Use this to run
    control-plane tasks on the login node itself (e.g. dask-scheduler-up) without SSH.
    """
    import os

    try:
        subscription_id = os.environ['SUBSCRIPTION']
        resource_group = os.environ['CI_RESOURCE_GROUP']
        workspace_name = os.environ['CI_WORKSPACE']
    except Exception as error:
        print("please set the export variables: SUBSCRIPTION, CI_RESOURCE_GROUP and CI_WORKSPACE")
        exit(-1)

    pwd = os.environ['PWD']

    from azure.ai.ml import MLClient, command
    from azure.ai.ml.entities import Environment
    from azure.identity import DefaultAzureCredential
    import argparse

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

    parser = argparse.ArgumentParser(description='srun: run jobs on the login ComputeInstance')
    parser.prog = "srun"
    parser.add_argument('--container', default="None", type=str, help='container image for the job to run in')
    parser.add_argument('-e', '--environment', default="None", type=str, help='Azure Machine Learning environment, may use @latest')
    parser.add_argument('-p', '--partition', default="None", type=str,
                        help='ComputeInstance to run on. Defaults to auto-discovered login CI. Use <sinfo> to view partitions')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='provide output on found settings and job properties')
    parser.add_argument('-w', '--wrap', type=str, help='command line to be executed, should be enclosed with quotes')
    parser.add_argument('script', nargs='?', default="None", type=str, help='runscript to be executed')
    args = parser.parse_args(vargs)

    if (args.script == "None") and (args.wrap is None):
        print("Missing: provide either script to execute as argument or commandline to execute through --wrap option")
        exit(-1)

    if (args.script != "None") and (args.wrap is not None):
        print("Conflict: provide either script to execute as argument or commandline to execute through --wrap option")
        exit(-1)

    if (args.container != "None") and (args.environment != "None"):
        print("Conflict: provide either container or environment, cannot be used together")
        exit(-1)

    # A ComputeInstance is a valid command-job compute target. Auto-discover the login
    # CI (the single computeinstance) so users need not know its generated name.
    if (args.partition == "None"):
        instances = [c.name for c in ml_client.compute.list() if str(getattr(c, "type", "")).lower() == "computeinstance"]
        if len(instances) == 0:
            print("Missing: no ComputeInstance found in the workspace; provide one with -p")
            exit(-1)
        if len(instances) > 1:
            print("Ambiguous: multiple ComputeInstances found, provide one with -p: " + ", ".join(instances))
            exit(-1)
        args.partition = instances[0]
        if (args.verbose): print("auto-discovered login ComputeInstance: " + args.partition)

    if (args.container != "None"):
        if (args.verbose): print("using container: " + args.container)
        env_docker_image = Environment(
                image=args.container,
                name="srun-container-image",
                description="Environment created from a Docker image.",
                )
        ml_client.environments.create_or_update(env_docker_image)
        args.environment = "srun-container-image@latest"
        if (args.verbose): print("created container environment: " + args.environment)

    if (args.environment == "None"):
        args.environment = "amlhpc-ubuntu2204@latest"
        if (args.verbose): print("using default environment: " + args.environment)

    if (args.verbose): print("using environment: " + args.environment)

    if (args.script != "None"):
        job_code = pwd + "/" + args.script
        job_command = "chmod +x " + args.script + "; " + args.script
        if (args.verbose):
            print("provided script to be uploaded: " + job_code)
            print("initial command (through runscript): " + job_command)

    if (args.wrap is not None):
        job_code = None
        job_command = args.wrap
        if (args.verbose):
            print("no script to be uploaded.")
            print("initial command (through wrapped script): " + job_command)

    command_job = command(
        code=job_code,
        command=job_command,
        environment=args.environment,
        instance_count=1,
        compute=args.partition,
        )

    returned_job = ml_client.jobs.create_or_update(command_job)
    print(returned_job.name)
