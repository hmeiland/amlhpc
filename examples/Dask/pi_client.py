"""Distributed Monte Carlo estimate of pi, run against an amlhpc Dask cluster.

Connects to the scheduler at $DASK_SCHEDULER (e.g. tcp://10.0.1.4:8786 as printed
by the scheduler job), waits for at least one worker, then fans out embarrassingly
parallel dart-throwing tasks and reduces the counts into a pi estimate you can check.

Run it as a CI job via srun (see README section 4).
"""
import os
import random

from dask.distributed import Client


def count_inside(n_samples, seed):
    rng = random.Random(seed)
    inside = 0
    for _ in range(n_samples):
        x, y = rng.random(), rng.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return inside


def main():
    scheduler = os.environ.get("DASK_SCHEDULER")
    if not scheduler:
        raise SystemExit("set DASK_SCHEDULER, e.g. tcp://10.0.1.4:8786 (from the scheduler job)")

    client = Client(scheduler)
    print("CLIENT_CONNECTED", flush=True)
    client.wait_for_workers(1, timeout=600)
    print("NWORKERS=%d" % len(client.scheduler_info()["workers"]), flush=True)

    n_tasks = 512
    samples_per_task = 2_000_000
    total_samples = n_tasks * samples_per_task

    futures = [client.submit(count_inside, samples_per_task, seed) for seed in range(n_tasks)]
    inside_total = sum(client.gather(futures))
    pi_estimate = 4.0 * inside_total / total_samples
    err = abs(pi_estimate - 3.141592653589793)

    print("SAMPLES=%d" % total_samples, flush=True)
    print("PI=%.10f" % pi_estimate, flush=True)
    print("ERROR=%.3e" % err, flush=True)
    print("RESULT_OK" if err < 1e-2 else "RESULT_BAD", flush=True)
    client.close()


if __name__ == "__main__":
    main()
