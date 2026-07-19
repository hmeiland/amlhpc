"""Tests for job cancellation (amlhpc/jobcontrol.py:cancel_job).

``cancel_job`` backs the Slurm ``scancel``, PBS ``qdel`` and LSF ``bkill``
commands. It parses one-or-more JOBIDs, then calls ``begin_cancel`` on the
AML client for each. We stub ``get_ml_client`` so no Azure connection is made
and assert on the argv parsing, per-job messaging, and exit codes.
"""

import pytest
from azure.core.exceptions import ResourceNotFoundError

import amlhpc.jobcontrol as jobcontrol
from amlhpc.jobcontrol import cancel_job


class FakeJobs:
    def __init__(self, missing=(), failing=()):
        self.cancelled = []
        self._missing = set(missing)
        self._failing = set(failing)

    def begin_cancel(self, jobid):
        if jobid in self._missing:
            raise ResourceNotFoundError("not found")
        if jobid in self._failing:
            raise RuntimeError("boom")
        self.cancelled.append(jobid)


class FakeMLClient:
    def __init__(self, **kw):
        self.jobs = FakeJobs(**kw)


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeMLClient()

    def factory():
        return client

    monkeypatch.setattr(jobcontrol, "get_ml_client", factory)
    return client


def test_requires_at_least_one_jobid(fake_client):
    # argparse enforces nargs='+' -> exits non-zero when no JOBID given
    with pytest.raises(SystemExit) as exc:
        cancel_job("scancel", [])
    assert exc.value.code != 0


def test_single_job_cancelled(fake_client, capsys):
    cancel_job("scancel", ["job_abc"])
    assert fake_client.jobs.cancelled == ["job_abc"]
    out = capsys.readouterr().out
    assert "scancel: cancellation requested for job 'job_abc'" in out


def test_multiple_jobs_cancelled_and_prog_name_used(fake_client, capsys):
    cancel_job("qdel", ["job1", "job2", "job3"])
    assert fake_client.jobs.cancelled == ["job1", "job2", "job3"]
    out = capsys.readouterr().out
    # the prog label (qdel/bkill/scancel) is reflected in the message
    assert out.count("qdel: cancellation requested") == 3


def test_missing_job_reports_and_exits_nonzero(monkeypatch, capsys):
    client = FakeMLClient(missing={"ghost"})
    monkeypatch.setattr(jobcontrol, "get_ml_client", lambda: client)
    with pytest.raises(SystemExit) as exc:
        cancel_job("bkill", ["ghost"])
    assert exc.value.code == 1
    assert "bkill: job 'ghost' not found" in capsys.readouterr().out


def test_partial_failure_still_cancels_others_and_exits_nonzero(monkeypatch, capsys):
    client = FakeMLClient(missing={"ghost"})
    monkeypatch.setattr(jobcontrol, "get_ml_client", lambda: client)
    with pytest.raises(SystemExit) as exc:
        cancel_job("scancel", ["ok1", "ghost", "ok2"])
    assert exc.value.code == 1
    # the good ones are still cancelled despite the missing one
    assert client.jobs.cancelled == ["ok1", "ok2"]


def test_generic_error_reports_and_exits_nonzero(monkeypatch, capsys):
    client = FakeMLClient(failing={"stuck"})
    monkeypatch.setattr(jobcontrol, "get_ml_client", lambda: client)
    with pytest.raises(SystemExit) as exc:
        cancel_job("scancel", ["stuck"])
    assert exc.value.code == 1
    assert "failed to cancel job 'stuck'" in capsys.readouterr().out


def test_scancel_wrapper_delegates_to_cancel_job(monkeypatch):
    from amlhpc.slurm.scancel import scancel

    seen = {}
    monkeypatch.setattr(
        "amlhpc.jobcontrol.cancel_job",
        lambda prog, vargs: seen.update(prog=prog, vargs=vargs),
    )
    scancel(["wheat_sand_gr2xcdpl2w"])
    assert seen == {"prog": "scancel", "vargs": ["wheat_sand_gr2xcdpl2w"]}
