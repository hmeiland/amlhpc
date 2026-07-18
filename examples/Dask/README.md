# Dask on amlhpc

Run a distributed Dask cluster on Azure Machine Learning using the built-in amlhpc dask commands:

- `dask-scheduler-up` — run a Dask scheduler on this Compute Instance (CI)
- `dask-up` — submit a pool of AmlCompute worker VMs that connect back to the scheduler
- `dask-down` — cancel the worker pool so the VMs scale to zero

The scheduler binds to the CI's VNet-private IP (auto-detected via IMDS) and prints a `tcp://HOST:8786`
address. Both the worker pool (`dask-up --scheduler`) and your notebook (`Client("tcp://HOST:8786")`)
connect to that same address. Everything runs inside the workspace VNet, so no hardcoded IPs and no
manual worker scripts are needed.

Choose the **Python 3.8 - AzureML** kernel for the notebook, since dask is already installed there.

## Connect your terminals to the workspace

`dask-scheduler-up`, `dask-up` and `dask-down` need the workspace environment variables
(`SUBSCRIPTION`, `CI_RESOURCE_GROUP`, `CI_WORKSPACE`). If you deployed the cluster with `deploy init`,
those exports were written to `~/.amlhpc/<clustername>.sh`. In **every** terminal you open, source the
profile first, then confirm the connection:
```bash
source ~/.amlhpc/<clustername>.sh
sinfo
```
`sinfo` should list your partitions (e.g. `f4s`). Inside an Azure Machine Learning Compute Instance,
`CI_RESOURCE_GROUP` and `CI_WORKSPACE` are usually already set, so sourcing the profile mainly restores
`SUBSCRIPTION` — but sourcing it is the reliable way to target a specific cluster.

## 1. Start the scheduler (terminal 1, on the CI)

```bash
dask-scheduler-up
```
This stays in the foreground and prints the address the workers and client both use:
```
scheduler address (use for dask-up --scheduler and Client): tcp://10.0.1.5:8786
dashboard: http://10.0.1.5:8787
```
Leave this terminal running. Copy the `tcp://…:8786` address for the next steps.

## 2. Add worker VMs (terminal 2, on the CI)

Submit a pool of 4 worker VMs on the `f4s` partition, all attached to the scheduler and tagged with a
session id so you can tear them down later:
```bash
dask-up -p f4s -s tcp://10.0.1.5:8786 --session pi-demo -N 4
```
`dask-up` prints one line — `jobname  session  pool  nodes` — and returns immediately; the AmlCompute
job stays *Running* for the life of the session while the workers run in the foreground. The workers
auto-install a matching `dask`/`distributed` version if the image differs, so no image rebuild is
needed. Watch them register in the client widget (or the Dask dashboard) before running the calculation.

> To add more capacity, run `dask-up` again against another partition, reusing the same
> `--scheduler` and `--session`, e.g. `dask-up -p f16s -s tcp://10.0.1.5:8786 --session pi-demo -N 8`.

## 3. A real calculation: estimate π with distributed Monte Carlo

From the notebook, connect a client to the running scheduler by its address:
```python
from dask.distributed import Client, progress
client = Client("tcp://10.0.1.5:8786")   # the address dask-scheduler-up printed
client
```

Throw random darts at the unit square; the fraction landing inside the quarter circle approximates π/4.
The work is embarrassingly parallel, so each task runs independently on a worker VM and we reduce the
counts. More samples (and more workers) give a closer estimate — a result you can actually check.

Define the per-task worker function:
```python
import random

def count_inside(n_samples, seed):
    rng = random.Random(seed)
    inside = 0
    for _ in range(n_samples):
        x, y = rng.random(), rng.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return inside
```

Then fan the tasks out across the cluster, gather the counts, and compute π:
```python
%%time

n_tasks = 512
samples_per_task = 2_000_000
total_samples = n_tasks * samples_per_task

futures = [client.submit(count_inside, samples_per_task, seed) for seed in range(n_tasks)]
progress(futures)

inside_total = sum(client.gather(futures))
pi_estimate = 4.0 * inside_total / total_samples

print(f"samples : {total_samples:,}")
print(f"pi ~    : {pi_estimate}")
print(f"error   : {abs(pi_estimate - 3.141592653589793):.2e}")
```

With ~1e9 samples the estimate typically lands within a few 1e-4 of π. In the Dask dashboard — the link
is under Compute -> Applications (you may have to press the 3 dots to expand), or the
`http://HOST:8787` address the scheduler printed — you can watch the tasks being distributed across the
worker VMs:
![Dask Dashboard](dask_dashboard.png)

## 4. Tear down

When you are done, disconnect the client and stop the worker session. `dask-down` cancels every worker
pool tagged with the session id; the AmlCompute nodes then go idle and scale to zero, so you stop paying
for them:
```python
client.close()   # in the notebook
```
```bash
dask-down --session pi-demo    # terminal 2
```
Finally, stop the scheduler by pressing `Ctrl-C` in terminal 1.
