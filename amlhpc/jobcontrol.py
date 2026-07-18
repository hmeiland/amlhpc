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


def _job_times(job):
    """Best-effort (start, end) datetimes for a job, or (None, None).

    AML exposes lifecycle timestamps under creation_context and/or the
    properties bag depending on job type and SDK version; probe both.
    """
    from datetime import datetime, timezone

    def _parse(val):
        if val is None:
            return None
        if isinstance(val, datetime):
            return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
        try:
            s = str(val).replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    props = getattr(job, "properties", None) or {}
    cc = getattr(job, "creation_context", None)

    start = (_parse(props.get("StartTimeUtc"))
             or _parse(props.get("startTimeUtc"))
             or _parse(getattr(cc, "created_at", None)))
    end = (_parse(props.get("EndTimeUtc"))
           or _parse(props.get("endTimeUtc"))
           or _parse(getattr(cc, "last_modified_at", None)))
    return start, end


def show_job_status(prog, vargs):
    """Show status of one or more jobs by JOBID. Backs Slurm sacct.

    Unlike squeue/qstat (which list every job), sacct targets specific
    JOBIDs and adds start/end timestamps from the job's lifecycle metadata.
    """
    import argparse

    parser = argparse.ArgumentParser(description=prog + ": accounting status for Azure Machine Learning jobs")
    parser.prog = prog
    parser.add_argument('jobid', nargs='+', help='one or more JOBIDs to report (see squeue/qstat/bjobs)')
    args = parser.parse_args(vargs)

    ml_client = get_ml_client()

    from azure.core.exceptions import ResourceNotFoundError

    header = ("JOBID" + " " * 27 + "NAME" + " " * 12 + "PARTITION" + " " * 7
              + "STATE" + " " * 11 + "START" + " " * 15 + "END")
    print(header)

    exit_code = 0
    for jobid in args.jobid:
        try:
            job = ml_client.jobs.get(jobid)
        except ResourceNotFoundError:
            print(prog + ": job '" + jobid + "' not found")
            exit_code = 1
            continue
        except Exception as error:
            print(prog + ": failed to get job '" + jobid + "': " + str(error))
            exit_code = 1
            continue

        start, end = _job_times(job)
        name = (job.name or "")[:31].ljust(32)
        display = str(getattr(job, "display_name", "") or "")[:15].ljust(16)
        compute = str(getattr(job, "compute", "") or "")[:15].ljust(16)
        state = str(getattr(job, "status", "") or "").ljust(16)
        start_s = (start.strftime("%Y-%m-%dT%H:%M:%S") if start else "-").ljust(20)
        end_s = end.strftime("%Y-%m-%dT%H:%M:%S") if end else "-"
        print(name + display + compute + state + start_s + end_s)

    if exit_code:
        exit(exit_code)


def show_job_stats(prog, vargs):
    """Show node CPU/memory utilization for a running/finished job. Backs Slurm sstat.

    Utilization is an Azure Monitor platform metric on the *workspace* resource;
    it is NOT job-scoped (Azure Monitor exposes no RunId dimension for it), so
    the figures are read over the job's own time window and reflect the whole
    compute node - on a shared cluster that may include other jobs.
    """
    import argparse
    from datetime import timezone

    parser = argparse.ArgumentParser(
        description=prog + ": node CPU/memory utilization for an Azure Machine Learning job")
    parser.prog = prog
    parser.add_argument('jobid', help='JOBID to report (see squeue/qstat/sacct)')
    parser.add_argument('--history', action='store_true',
                        help='print the full per-minute time series instead of just the latest sample')
    args = parser.parse_args(vargs)

    ml_client = get_ml_client()

    from azure.core.exceptions import ResourceNotFoundError

    try:
        job = ml_client.jobs.get(args.jobid)
    except ResourceNotFoundError:
        print(prog + ": job '" + args.jobid + "' not found")
        exit(1)

    start, end = _job_times(job)
    if start is None:
        print(prog + ": job '" + args.jobid + "' has no start time yet (not running)")
        exit(1)

    from datetime import datetime, timedelta
    if end is None:
        end = datetime.now(timezone.utc)
    start = start - timedelta(minutes=1)
    end = end + timedelta(minutes=1)

    import os
    subscription_id = os.environ['SUBSCRIPTION']
    resource_group = os.environ['CI_RESOURCE_GROUP']
    workspace_name = os.environ['CI_WORKSPACE']
    workspace_uri = (
        "/subscriptions/" + subscription_id
        + "/resourceGroups/" + resource_group
        + "/providers/Microsoft.MachineLearningServices/workspaces/" + workspace_name
    )

    try:
        from azure.monitor.query import MetricsQueryClient, MetricAggregationType
        from azure.identity import DefaultAzureCredential
    except ImportError:
        print(prog + ": requires the 'azure-monitor-query' package (pip install 'azure-monitor-query<2')")
        exit(1)

    metric_names = ["CpuUtilizationPercentage", "CpuMemoryUtilizationPercentage"]
    client = MetricsQueryClient(DefaultAzureCredential())
    response = client.query_resource(
        workspace_uri,
        metric_names=metric_names,
        timespan=(start, end),
        granularity=timedelta(minutes=1),
        aggregations=[MetricAggregationType.AVERAGE, MetricAggregationType.MAXIMUM],
    )

    print("job " + args.jobid + " node utilization (whole-node, over job window; not job-scoped)")
    for metric in response.metrics:
        points = []
        for ts in metric.timeseries:
            for d in ts.data:
                if d.average is not None or d.maximum is not None:
                    points.append((d.timestamp, d.average, d.maximum))
        if not points:
            print("  " + metric.name + ": no data")
            continue
        if args.history:
            print("  " + metric.name + " (avg/max per minute):")
            for tstamp, avg, mx in points:
                print("    " + str(tstamp)[:19] + "  avg=" + _fmt_pct(avg) + "  max=" + _fmt_pct(mx))
        else:
            latest = points[-1]
            peak = max(points, key=lambda p: (p[2] if p[2] is not None else -1))
            print("  " + metric.name + ": latest avg=" + _fmt_pct(latest[1])
                  + "  peak max=" + _fmt_pct(peak[2]) + " @ " + str(peak[0])[:19]
                  + "  (" + str(len(points)) + " samples)")


def _fmt_pct(v):
    if v is None:
        return "-"
    return format(v, ".1f") + "%"


def attach_job(prog, vargs):
    """Attach to a job's output. Backs Slurm sattach.

    Default: one-shot tail of the job's std_log. With -f/--follow: stream the
    log until the job finishes (via the SDK's stream()).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=prog + ": show or follow the log output of an Azure Machine Learning job")
    parser.prog = prog
    parser.add_argument('jobid', help='JOBID to attach to (see squeue/qstat/sacct)')
    parser.add_argument('-f', '--follow', action='store_true',
                        help='stream the log until the job completes (like tail -f)')
    args = parser.parse_args(vargs)

    ml_client = get_ml_client()

    from azure.core.exceptions import ResourceNotFoundError

    try:
        ml_client.jobs.get(args.jobid)
    except ResourceNotFoundError:
        print(prog + ": job '" + args.jobid + "' not found")
        exit(1)

    if args.follow:
        try:
            ml_client.jobs.stream(args.jobid)
        except Exception as error:
            print(prog + ": failed to stream job '" + args.jobid + "': " + str(error))
            exit(1)
        return

    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        try:
            ml_client.jobs.download(args.jobid, download_path=tmp)
        except Exception as error:
            print(prog + ": failed to download logs for job '" + args.jobid + "': " + str(error))
            exit(1)

        log_path = None
        for root, _dirs, files in os.walk(tmp):
            for fname in files:
                if fname == "std_log.txt":
                    log_path = os.path.join(root, fname)
                    break
            if log_path:
                break

        if not log_path:
            print(prog + ": no std_log.txt found for job '" + args.jobid + "' (job may not have produced output yet)")
            exit(1)

        with open(log_path, "r") as fh:
            print(fh.read(), end="")


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
