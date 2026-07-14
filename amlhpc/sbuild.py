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


def sbuild(vargs=None):
    import os

    try:
        subscription_id = os.environ['SUBSCRIPTION']
        resource_group = os.environ['CI_RESOURCE_GROUP']
        workspace_name = os.environ['CI_WORKSPACE']
    except Exception as error:
        print("please set the export variables: SUBSCRIPTION, CI_RESOURCE_GROUP and CI_WORKSPACE")
        exit(-1)

    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
    from azure.ai.ml.entities import Environment, BuildContext
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

    parser = argparse.ArgumentParser(description='sbuild: build amlhpc container environments in Azure ML. Uses the AML build workflow, which provisions the workspace container registry if required.')
    parser.prog = "sbuild"
    parser.add_argument('-p', '--path', default="environments", type=str,
                        help='directory holding environment build contexts, each a subdirectory containing a Dockerfile')
    parser.add_argument('-e', '--environment', default="None", type=str,
                        help='build only this environment (subdirectory name); default builds every context found')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='provide output on found settings and build properties')
    args = parser.parse_args(vargs)

    if not os.path.isdir(args.path):
        print("Missing: environments directory '" + args.path + "' not found; run from a checkout or set --path")
        exit(-1)

    if (args.environment != "None"):
        environment_list = [args.environment]
    else:
        environment_list = sorted([d for d in os.listdir(args.path) if os.path.isfile(os.path.join(args.path, d, "Dockerfile"))])

    if not environment_list:
        print("Missing: no build contexts (subdirectory with a Dockerfile) found under '" + args.path + "'")
        exit(-1)

    for name in environment_list:
        context = os.path.join(args.path, name)
        if not os.path.isfile(os.path.join(context, "Dockerfile")):
            print("skipping " + name + ": no Dockerfile found in " + context)
            continue
        if (args.verbose):
            print("building environment '" + name + "' from context " + context)
        environment = Environment(
            name=name,
            build=BuildContext(path=context),
            )
        returned_environment = ml_client.environments.create_or_update(environment)
        print(returned_environment.name + ":" + str(returned_environment.version))
