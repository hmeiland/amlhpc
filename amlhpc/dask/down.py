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


def dask_down(vargs=None):
    """Tear down a Dask worker session by cancelling every worker pool tagged with its id.

    Cancelling each command job stops the foreground dask worker, after which the
    AmlCompute nodes go idle and scale to zero on their configured idle timeout.
    """
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

    parser = argparse.ArgumentParser(description='dask-down: stop a Dask worker session and let its AmlCompute nodes scale to zero')
    parser.prog = "dask-down"
    parser.add_argument('--session', required=True, type=str, help='session id whose worker pools should be cancelled')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='provide output on matched jobs')
    args = parser.parse_args(vargs)

    matched = 0
    for job in ml_client.jobs.list():
        tags = getattr(job, 'tags', None) or {}
        if tags.get('dask_session') == args.session and tags.get('dask_role') == 'worker':
            matched += 1
            if (args.verbose):
                print("cancelling " + job.name + " (pool " + str(tags.get('dask_pool')) + ")")
            ml_client.jobs.begin_cancel(job.name)
            print(job.name + "\tcancelled")

    if matched == 0:
        print("no active worker pools found for session '" + args.session + "'")
        exit(-1)

    print("requested cancel for " + str(matched) + " pool(s); nodes will scale down after their idle timeout")
