# Dask on amlhpc

Run a persistent Dask **scheduler** on the login `ComputeInstance` and attach one or more
`AmlCompute` clusters as **worker pools**. This mirrors the existing amlhpc model: the
ComputeInstance is the head node, and each `AmlCompute` cluster (a "partition" in `sinfo`)
becomes an elastic pool of Dask workers.

```
                 ComputeInstance (login VM, always-on)
                 ┌─────────────────────────────────────┐
                 │  dask scheduler   :8786  (TCP)        │
                 │  dashboard        :8787  (HTTP)       │
                 │  Client("tcp://<ci-ip>:8786")         │
                 └───────────────▲───────────▲───────────┘
                    tcp 8786 +    │           │   tcp 8786 +
                    callback      │           │   callback
                    30000-30100   │           │   30000-30100
          ┌───────────────────────┴──┐   ┌────┴──────────────────────┐
          │ AmlCompute pool "cpu"     │   │ AmlCompute pool "gpu"     │
          │ command job, N nodes      │   │ command job, M nodes      │
          │ each: dask worker …       │   │ each: dask worker …       │
          └───────────────────────────┘   └───────────────────────────┘
```

## Core design decisions

These follow directly from how AML command jobs and Dask actually behave (see rationale
at the bottom). They are non-negotiable for a working deployment.

1. **Worker pools are long-running, session-scoped AML jobs — not batch jobs.**
   A `dask worker` runs in the **foreground until stopped**. The AML command job stays
   `Running` (never `Succeeded`) for the life of the Dask session. This is a legitimate
   use of a command job — AML only requires the command to keep running, not to be short.

2. **Never background/daemonize the worker.** No `nohup`, `&`, `disown`. If the command
   exits, AML considers the job done and releases the node. The `dask worker` process
   *is* the job's foreground command.

3. **Disable retries on worker jobs.** A retry can resurrect a worker pool after the
   scheduler/session is gone — worse than a dead pool.

4. **Discover the scheduler IP at submit time** on the ComputeInstance and pass it to each
   worker job as `tcp://<ci-private-ip>:8786`. Do not rely on DNS for the MVP.

5. **Pin ports and advertised addresses explicitly.** The single most likely failure is
   Dask advertising an address/port the scheduler cannot dial back. Bind the scheduler on
   the CI private interface; make workers advertise their AmlCompute private IP with a
   fixed worker-port range.

6. **"Idle Dask" ≠ "idle AML".** A worker waiting for tasks is still a billing AML node.
   AmlCompute idle-scaledown does **not** fire while a `dask worker` command runs. Cost is
   controlled only by explicit teardown (`dask-down`) or an optional lifetime/TTL.

## Networking invariant

Dask connections are **bidirectional**: workers dial the scheduler on 8786 to register,
then the scheduler opens long-lived connections **back** to each worker's listening port.
Same VNet/subnet (`10.0.1.0/24` in `amlhpc_simple.bicep`) satisfies this **only if
east-west VNet traffic is allowed**. `enableNodePublicIp: false` and
`remoteLoginPortPublicAccess: Disabled` do **not** block private intra-VNet ports.

Required, assuming a hardened NSG:

| Direction                  | Port         | Purpose                         |
|----------------------------|--------------|---------------------------------|
| worker → scheduler (CI)    | tcp 8786     | worker registration + RPC       |
| scheduler (CI) → worker    | tcp 30000-30100 | task push, data, callbacks   |
| (local/tunnel only)        | tcp 8787     | scheduler dashboard             |

If the subnet already allows `VirtualNetwork`→`VirtualNetwork` on all ports, defaults work
and the fixed range is just hardening insurance.

## Commands (proposed CLI, fits the amlhpc entry-point pattern)

New console entry points alongside `sbatch`/`sinfo`/`squeue`. They are **session-oriented**
(`dask-up`/`dask-down`), not `sbatch`-style batch, because worker jobs are intentionally
long-lived and tagged.

### `dask-scheduler-up` (run on the ComputeInstance)
Starts the scheduler in the foreground on the CI, bound to the CI private interface:

```bash
CI_IP=$(curl -s -H Metadata:true \
  "http://169.254.169.254/metadata/instance/network/interface/0/ipv4/ipAddress/0/privateIpAddress?api-version=2021-02-01&format=text")

dask scheduler \
  --host "$CI_IP" \
  --port 8786 \
  --dashboard-address "$CI_IP:8787"
```

Print `tcp://$CI_IP:8786` — that's the address every pool and client uses.

### `dask-up` (submit a worker pool to a partition)
Submits an AML command job to an `AmlCompute` cluster. Conceptually:

```
dask-up --pool cpu --nodes 8  --scheduler tcp://<ci-ip>:8786 --session <id>
dask-up --pool gpu --nodes 2  --scheduler tcp://<ci-ip>:8786 --session <id> \
        --nthreads 2 --resources "GPU=1"
```

Each node runs this **foreground** worker command (built by `dask-up`, submitted via
`ml_client.jobs.create_or_update`):

```bash
WORKER_IP=$(curl -s -H Metadata:true \
  "http://169.254.169.254/metadata/instance/network/interface/0/ipv4/ipAddress/0/privateIpAddress?api-version=2021-02-01&format=text")

exec dask worker "$DASK_SCHEDULER" \
  --host "$WORKER_IP" \
  --worker-port 30000:30100 \
  --nworkers "$DASK_NWORKERS" \
  --nthreads "$DASK_NTHREADS" \
  --name "${DASK_SESSION}-${DASK_POOL}" \
  --no-dashboard \
  ${DASK_RESOURCES:+--resources "$DASK_RESOURCES"}
```

Job construction requirements (in the amlhpc submit code):
- `command = <above>`, `compute = <pool/partition name>`, `instance_count = --nodes`.
- Pass `DASK_SCHEDULER`, `DASK_SESSION`, `DASK_POOL`, `DASK_NWORKERS`, `DASK_NTHREADS`,
  `DASK_RESOURCES` via `environment_variables`.
- **Tag the job** with `dask_session=<id>`, `dask_pool=<pool>` for teardown lookup.
- **Disable retries** (set job/component retry to 0).
- `exec` so `dask worker` becomes PID 1's foreground child and receives SIGTERM on cancel.

Add a pool by simply running `dask-up` again against a different cluster with the same
`--scheduler`/`--session`. All pools register to the one scheduler (Dask supports many
independent worker groups → one scheduler with no limit).

### Client usage (on the ComputeInstance)
```python
from distributed import Client
client = Client("tcp://<ci-ip>:8786")
```

### `dask-down` (teardown)
```
dask-down --session <id>
```
1. Best-effort graceful worker retirement via the scheduler
   (`Client.retire_workers()` / `client.shutdown()` if killing the whole session).
2. `ml_client.jobs.begin_cancel(...)` every job tagged `dask_session=<id>`.
3. Cancelling the job stops the foreground `dask worker`; nodes go idle and AmlCompute
   scales to zero after `nodeIdleTimeBeforeScaleDown` (PT120S).

## Scaling model

**MVP: one fixed-size worker job per cluster** (`dask-up --pool X --nodes N`). Simple,
predictable, matches AML's allocation model.

**Dask adaptive (`cluster.adapt()`) is a later feature, and complementary — not redundant.**
AmlCompute autoscale sizes a cluster to satisfy `instance_count`, but it **cannot** grow or
shrink the node count *inside an already-running worker job* based on Dask task pressure.
True adaptivity requires a custom AML-backed cluster manager that submits/cancels `dask-up`
jobs in response to load — handling slow provisioning, preemption, and oscillation. Defer it.

## Cost & lifecycle warnings

- Scheduler on the CI is always-on — that's fine, it's the login VM you already pay for.
- **Worker nodes bill for the entire Dask session, idle or not.** Always `dask-down`.
- Consider an optional `--lifetime` on `dask-up` (maps to `dask worker --lifetime SECONDS`)
  as a safety TTL so forgotten pools self-terminate.
- **If the scheduler process dies/restarts, existing worker pools are stale** — tear them
  down and recreate rather than trying to reconnect.

## Preflight (recommended before first real run)

A tiny connectivity check job that, from a worker node, opens a TCP socket to
`tcp://<ci-ip>:8786` and has the scheduler dial back to the worker's `30000-30100` range.
This catches NSG/address-advertisement problems before you burn time launching real pools.

---

### Rationale (verified)

**Dask facts** (official distributed docs): scheduler defaults to `:8786` (TCP) / `:8787`
(dashboard); workers register via `tcp://SCHEDULER:8786`; the scheduler opens connections
**back** to workers (bidirectional); many independent worker groups can join one scheduler;
`Client("tcp://host:8786")` is the exact client syntax; `--worker-port START:END` pins the
listening range; `cluster.adapt()` works only for managed Cluster objects, **not**
CLI-launched workers.

**AML facts** (existing amlhpc code): `ComputeInstance` is a single persistent VM;
`AmlCompute` is a min-0 autoscaling cluster; `sbatch` builds a `command` job with
`instance_count` + `compute=<partition>` and submits via `ml_client.jobs.create_or_update`;
multi-node jobs already share intra-job networking. A command job stays `Running` as long
as its foreground command runs — which is exactly what makes a long-lived worker pool viable.
