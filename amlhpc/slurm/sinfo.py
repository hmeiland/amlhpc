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


def sinfo(vargs=None):
    import os
    try:
        subscription_id = os.environ['SUBSCRIPTION']
        resource_group = os.environ['CI_RESOURCE_GROUP']
        workspace_name = os.environ['CI_WORKSPACE']
    except Exception as error:
        print("please set the export variables: SUBSCRIPTION, CI_RESOURCE_GROUP, and CI_WORKSPACE")
        exit(-1)

    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
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
        )

    sinfo_list = ml_client.compute.list()

    print("PARTITION\tAVAIL\tVM_SIZE\t\t\tNODES\tSTATE")
    for i in sinfo_list:
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
