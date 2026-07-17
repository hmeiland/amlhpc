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


# Worker command run on every AmlCompute node. It must run in the FOREGROUND:
# if it exits, AML considers the job done and releases the node. Each worker
# advertises its own VNet-routable private IP (via IMDS) on a pinned port range
# so the scheduler can dial back (Dask connections are bidirectional).
# "dask" is resolved next to the active python: in a conda-materialized image
# env its bin dir is not always on the container's PATH. Dask requires the
# scheduler and workers to run the SAME distributed version; dask-up captures
# the scheduler-side version and the worker pip-installs to match if the image
# has a different (or no) distributed, so dask-up works without an image rebuild.
_WORKER_SCRIPT = r'''set -e
WORKER_IP=$(curl -s -H Metadata:true "http://169.254.169.254/metadata/instance/network/interface/0/ipv4/ipAddress/0/privateIpAddress?api-version=2021-02-01&format=text")
echo "dask worker private ip: ${WORKER_IP}"
echo "connecting to scheduler: ${DASK_SCHEDULER}"
RESOURCE_ARGS=""
if [ -n "${DASK_RESOURCES}" ]; then RESOURCE_ARGS="--resources ${DASK_RESOURCES}"; fi
LIFETIME_ARGS=""
if [ -n "${DASK_LIFETIME}" ]; then LIFETIME_ARGS="--lifetime ${DASK_LIFETIME}"; fi
HAVE_VER=$(python -c "import distributed; print(distributed.__version__)" 2>/dev/null || true)
if [ "${HAVE_VER}" != "${DASK_DISTRIBUTED_VERSION}" ]; then
  echo "worker distributed=${HAVE_VER:-none}, scheduler wants ${DASK_DISTRIBUTED_VERSION}; installing to match"
  python -m pip install --no-cache-dir "distributed==${DASK_DISTRIBUTED_VERSION}" "dask==${DASK_DISTRIBUTED_VERSION}"
fi
export PATH="$(python -m site --user-base)/bin:$(dirname "$(command -v python)"):${PATH}"
DASK_BIN="$(command -v dask || true)"
if [ -z "${DASK_BIN}" ]; then DASK_BIN="$(python -m site --user-base)/bin/dask"; fi
exec "${DASK_BIN}" worker "${DASK_SCHEDULER}" \
  --host "${WORKER_IP}" \
  --worker-port ${DASK_WORKER_PORTS} \
  --nworkers ${DASK_NWORKERS} \
  --nthreads ${DASK_NTHREADS} \
  --name "${DASK_SESSION}-${DASK_POOL}" \
  --no-dashboard \
  ${RESOURCE_ARGS} ${LIFETIME_ARGS}
'''


def dask_up(vargs=None):
    """Submit a Dask worker pool as a long-running AML command job to an AmlCompute cluster.

    The job stays Running for the life of the Dask session (workers run in the foreground).
    Jobs are tagged with the session id so dask-down can find and cancel them, and retries
    are disabled so a dead pool is never resurrected. Run again against another cluster,
    reusing --scheduler/--session, to add more pools to the same scheduler.
    """
    import os

    try:
        subscription_id = os.environ['SUBSCRIPTION']
        resource_group = os.environ['CI_RESOURCE_GROUP']
        workspace_name = os.environ['CI_WORKSPACE']
    except Exception as error:
        print("please set the export variables: SUBSCRIPTION, CI_RESOURCE_GROUP and CI_WORKSPACE")
        exit(-1)

    from azure.ai.ml import MLClient, command
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

    parser = argparse.ArgumentParser(description='dask-up: launch a Dask worker pool on an AmlCompute cluster')
    parser.prog = "dask-up"
    parser.add_argument('-p', '--pool', required=True, type=str,
                        help='compute partition (AmlCompute cluster) to run workers on. Use <sinfo> to view partitions')
    parser.add_argument('-s', '--scheduler', required=True, type=str,
                        help='scheduler address, e.g. tcp://<compute-instance-ip>:8786 (printed by dask-scheduler-up)')
    parser.add_argument('--session', default="None", type=str,
                        help='session id tying this pool to a scheduler session (default: auto-generated)')
    parser.add_argument('-N', '--nodes', default=1, type=int, help='number of worker VMs to allocate (default: 1)')
    parser.add_argument('--nworkers', default=1, type=int, help='worker processes per node (default: 1)')
    parser.add_argument('--nthreads', default=0, type=int, help='threads per worker (default: 0 = auto)')
    parser.add_argument('--worker-ports', default="30000:30100", type=str,
                        help='pinned worker listening port range for scheduler callbacks (default: 30000:30100)')
    parser.add_argument('--resources', default="None", type=str,
                        help='Dask worker resource labels, e.g. "GPU=1" (quote if multiple)')
    parser.add_argument('--lifetime', default="None", type=str,
                        help='auto-shutdown each worker after N seconds (safety TTL; default: none)')
    parser.add_argument('--environment', default="amlhpc-ubuntu2204@latest", type=str,
                        help='AML environment (docker image) for the workers')
    parser.add_argument('--distributed-version', default="None", type=str,
                        help='dask/distributed version workers must match (default: auto-detect this CI scheduler version)')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='provide output on found settings and job properties')
    args = parser.parse_args(vargs)

    if not args.scheduler.startswith("tcp://"):
        print("Invalid: --scheduler must be a tcp:// address, e.g. tcp://10.0.1.4:8786")
        exit(-1)

    if args.session == "None":
        import uuid
        args.session = "dask-" + uuid.uuid4().hex[:8]
        if (args.verbose): print("generated session id: " + args.session)

    if args.distributed_version == "None":
        try:
            import distributed
            args.distributed_version = distributed.__version__
        except Exception:
            print("could not detect distributed version on this CI; pass --distributed-version to match the scheduler")
            exit(-1)
        if (args.verbose): print("scheduler distributed version: " + args.distributed_version)

    job_env = {
        "DASK_SCHEDULER": args.scheduler,
        "DASK_SESSION": args.session,
        "DASK_POOL": args.pool,
        "DASK_NWORKERS": str(args.nworkers),
        "DASK_NTHREADS": str(args.nthreads),
        "DASK_WORKER_PORTS": args.worker_ports,
        "DASK_RESOURCES": "" if args.resources == "None" else args.resources,
        "DASK_LIFETIME": "" if args.lifetime == "None" else args.lifetime,
        "DASK_DISTRIBUTED_VERSION": args.distributed_version,
    }

    if (args.verbose):
        print("pool (partition):   " + args.pool)
        print("scheduler:          " + args.scheduler)
        print("nodes:              " + str(args.nodes))
        print("workers per node:   " + str(args.nworkers))
        print("worker port range:  " + args.worker_ports)
        print("environment:        " + args.environment)

    worker_job = command(
        command=_WORKER_SCRIPT,
        environment=args.environment,
        instance_count=args.nodes,
        compute=args.pool,
        environment_variables=job_env,
        display_name=args.session + "-" + args.pool,
        tags={
            "dask_session": args.session,
            "dask_pool": args.pool,
            "dask_role": "worker",
        },
    )

    # A standalone command job runs once and is not retried by AML (retries are a
    # pipeline/batch-component concept), so a dead worker pool is never resurrected.
    returned_job = ml_client.jobs.create_or_update(worker_job)
    print(returned_job.name + "\t" + args.session + "\t" + args.pool + "\t" + str(args.nodes))
