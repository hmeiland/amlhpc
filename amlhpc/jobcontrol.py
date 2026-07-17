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


def get_ml_client():
    import os

    try:
        subscription_id = os.environ['SUBSCRIPTION']
        resource_group = os.environ['CI_RESOURCE_GROUP']
        workspace_name = os.environ['CI_WORKSPACE']
    except Exception:
        print("please set the export variables: SUBSCRIPTION, CI_RESOURCE_GROUP, and CI_WORKSPACE")
        exit(-1)

    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    import logging
    logging.getLogger('azure.ai.ml._utils').setLevel(logging.CRITICAL)

    try:
        on_aml = os.environ['APPSETTING_WEBSITE_SITE_NAME']
        if (on_aml == 'AMLComputeInstance'):
            credential = mlComputeAuth()
    except Exception:
        credential = DefaultAzureCredential()

    return MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
        )


def list_jobs(prog):
    """List jobs with their status. Backs PBS qstat and LSF bjobs."""
    ml_client = get_ml_client()

    header = "JOBID" + " " * 27 + "NAME" + " " * 12 + "PARTITION" + " " * 7 + "STATE"
    print(header)
    for page in ml_client.jobs.list().by_page():
        for job in page:
            name = (job.name or "")[:31].ljust(32)
            display = str(job.display_name or "")[:15].ljust(16)
            compute = str(job.compute or "")[:15].ljust(16)
            state = str(getattr(job, "status", "") or "")
            print(name + display + compute + state)


def cancel_job(prog, vargs):
    """Cancel one or more jobs by JOBID. Backs PBS qdel and LSF bkill."""
    import argparse

    parser = argparse.ArgumentParser(description=prog + ": cancel Azure Machine Learning jobs")
    parser.prog = prog
    parser.add_argument('jobid', nargs='+', help='one or more JOBIDs to cancel (see qstat/bjobs/squeue)')
    args = parser.parse_args(vargs)

    ml_client = get_ml_client()

    from azure.core.exceptions import ResourceNotFoundError

    exit_code = 0
    for jobid in args.jobid:
        try:
            ml_client.jobs.begin_cancel(jobid)
            print(prog + ": cancellation requested for job '" + jobid + "'")
        except ResourceNotFoundError:
            print(prog + ": job '" + jobid + "' not found")
            exit_code = 1
        except Exception as error:
            print(prog + ": failed to cancel job '" + jobid + "': " + str(error))
            exit_code = 1

    if exit_code:
        exit(exit_code)
