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
