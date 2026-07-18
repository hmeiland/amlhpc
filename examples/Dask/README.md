# Dask on amlhpc

Run a distributed Dask cluster on Azure Machine Learning entirely as CI jobs, driven from any
workstation that has the amlhpc CLI and an `az` login — no SSH, no CI terminal, no VNet access, and no
IMDS on your machine required. The scheduler and client run *as jobs on the login Compute Instance*, so
they share the CI network namespace: the scheduler binds the CI's VNet-private IP and the AmlCompute
workers (and the client) reach it over the workspace VNet.

> This is the exact sequence that was validated end-to-end on AML: an `f4s` worker computed
> π ≈ 3.14151 (error 7.9e-05) over ~1e9 samples, with the scheduler and client running as CI jobs.

The two amlhpc dask commands used are:

- `dask-up` — submit a pool of AmlCompute worker VMs that connect back to the scheduler
- `dask-down` — cancel the worker pool so the VMs scale to zero

(the scheduler itself is started with plain `dask scheduler` via `sbatch`, see below).

## 0. Prerequisites

Install amlhpc (0.6.0+ for `srun`) and connect your shell to the workspace. If you deployed the cluster
with `deploy init`, the workspace exports were written to `~/.amlhpc/<clustername>.sh`:
```bash
pip install --user amlhpc
source ~/.amlhpc/<clustername>.sh   # sets SUBSCRIPTION, CI_RESOURCE_GROUP, CI_WORKSPACE
sinfo                               # confirm partitions, e.g. f4s + the login-* CI partition
```
`sinfo` lists your worker partition (e.g. `f4s`) and the login ComputeInstance partition
(e.g. `login-ho7iiyqefiygoz`) — note both names.

## 1. Find the CI's private IP

Jobs on the CI can read it from IMDS; your workstation cannot, so ask the CI:
```bash
srun --wrap="curl -s -H Metadata:true 'http://169.254.169.254/metadata/instance/network/interface/0/ipv4/ipAddress/0/privateIpAddress?api-version=2021-02-01&format=text'"
az ml job stream -n <job-name>      # prints the IP, e.g. 10.0.1.4
```
Use that address (`10.0.1.4` below) for the scheduler bind, `dask-up --scheduler`, and the client.

## 2. Start the scheduler (a job on the login CI)

The default `amlhpc-ubuntu2204` environment does not ship dask, so pip-install it in the job. The `dask`
console script lands in `~/.local/bin`, which is **not** on the job's PATH — you must add it or
`dask scheduler` fails with `exec: dask: not found`:
```bash
sbatch -p <login-ci-partition> --wrap="pip install -q dask distributed; \
  export PATH=\"\$(python -m site --user-base)/bin:\$PATH\"; \
  exec dask scheduler --host 10.0.1.4 --port 8786 --no-dashboard"
az ml job stream -n <sched-job>     # wait for 'Scheduler at: tcp://10.0.1.4:8786'
```
The scheduler job stays *Running*; leave it up. Note the `distributed` version it reports — the workers
and client must match it (below it is `2026.7.1`).

## 3. Add worker VMs

Submit 4 worker VMs on the `f4s` partition, all attached to the scheduler and tagged with a session id
so you can tear them down later. Pin `--distributed-version` to the scheduler's version (your
workstation has no `distributed` to auto-detect from):
```bash
dask-up -p f4s -s tcp://10.0.1.4:8786 --session pi-demo -N 4 --distributed-version 2026.7.1
```
`dask-up` prints one line — `jobname  session  pool  nodes` — and returns immediately; the AmlCompute
job stays *Running* for the life of the session while the workers run in the foreground. Allocation of
the VMs takes a few minutes.

## 4. Run the client (a job on the login CI)

The client ([`pi_client.py`](pi_client.py)) connects to the scheduler, waits for workers, then runs a
distributed Monte Carlo estimate of π — embarrassingly parallel work whose result you can actually
check. `srun`'s script-mode uploads the file read-only (so `chmod +x` fails); drive it through `--wrap`
instead, base64-injecting the script so quoting is safe:
```bash
B64=$(base64 -w0 pi_client.py)
srun --wrap="pip install -q dask==2026.7.1 distributed==2026.7.1; echo $B64 | base64 -d > /tmp/pi_client.py; DASK_SCHEDULER=tcp://10.0.1.4:8786 python /tmp/pi_client.py"
az ml job stream -n <client-job>    # watch it connect and print the pi estimate
```
You should see, near the end of the stream:
```
NWORKERS=...
SAMPLES=1024000000
PI=3.14151...
ERROR=7.9e-05
RESULT_OK
```

## 5. Tear down

`dask-down` cancels every worker pool tagged with the session id; the AmlCompute nodes then go idle and
scale to zero, so you stop paying for them. There is no terminal to `Ctrl-C`, so also cancel the
scheduler job:
```bash
dask-down --session pi-demo
az ml job cancel -n <sched-job>
```
