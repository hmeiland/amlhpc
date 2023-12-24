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
