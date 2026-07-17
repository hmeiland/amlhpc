def _imds_private_ip():
    """Return this machine's primary private IPv4 via the Azure Instance Metadata Service."""
    import requests
    url = ("http://169.254.169.254/metadata/instance/network/interface/0/"
           "ipv4/ipAddress/0/privateIpAddress?api-version=2021-02-01&format=text")
    resp = requests.get(url, headers={'Metadata': 'true'}, timeout=5)
    resp.raise_for_status()
    return resp.text.strip()


def dask_scheduler_up(vargs=None):
    """Launch a Dask scheduler in the foreground on this ComputeInstance.

    Binds to the CI private interface so AmlCompute workers can reach it (and so the
    scheduler advertises a VNet-routable address for its callbacks). Prints the
    tcp:// address that every worker pool and client must use, then blocks.
    """
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(description='dask-scheduler-up: run a Dask scheduler on the ComputeInstance')
    parser.prog = "dask-scheduler-up"
    parser.add_argument('--host', default="None", type=str,
                        help='private IP/interface to bind (default: auto-detect via IMDS)')
    parser.add_argument('--port', default=8786, type=int, help='scheduler TCP port (default: 8786)')
    parser.add_argument('--dashboard-port', default=8787, type=int, help='dashboard HTTP port (default: 8787)')
    args = parser.parse_args(vargs)

    if args.host == "None":
        try:
            args.host = _imds_private_ip()
        except Exception as error:
            print("could not auto-detect private IP via IMDS; pass --host explicitly (" + str(error) + ")")
            exit(-1)

    scheduler_address = "tcp://" + args.host + ":" + str(args.port)
    print("scheduler address (use for dask-up --scheduler and Client): " + scheduler_address)
    print("dashboard: http://" + args.host + ":" + str(args.dashboard_port))

    # Resolve "dask" next to this interpreter: PATH lookup fails under a
    # non-login shell (tmux/nohup) where the conda env bin is not exported.
    dask_bin = os.path.join(os.path.dirname(sys.executable), "dask")
    if not os.path.exists(dask_bin):
        dask_bin = "dask"

    # Replace this process with the scheduler so signals reach it directly.
    os.execvp(dask_bin, [
        dask_bin, "scheduler",
        "--host", args.host,
        "--port", str(args.port),
        "--dashboard-address", args.host + ":" + str(args.dashboard_port),
    ])
