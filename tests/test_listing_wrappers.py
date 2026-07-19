"""Tests for job listing and the PBS/LSF thin wrappers.

``list_jobs`` (amlhpc/jobcontrol.py) backs ``qstat`` and ``bjobs``; it paginates
the AML job list and prints a fixed-width table. The ``qstat``/``qdel``/
``bjobs``/``bkill`` wrappers simply delegate to ``list_jobs``/``cancel_job``
with the right program label. All Azure access is stubbed.
"""

import pytest

import amlhpc.jobcontrol as jobcontrol
from amlhpc.jobcontrol import list_jobs
from amlhpc.pbs.qstat import qstat, qdel
from amlhpc.lsf.bjobs import bjobs, bkill


class FakeJob:
    def __init__(self, name, display_name, compute, status):
        self.name = name
        self.display_name = display_name
        self.compute = compute
        self.status = status


class FakePager:
    """Mimics ml_client.jobs.list() -> object with .by_page() -> iterable of pages."""

    def __init__(self, pages):
        self._pages = pages

    def by_page(self):
        return iter(self._pages)


class FakeJobsList:
    def __init__(self, pages):
        self._pages = pages

    def list(self):
        return FakePager(self._pages)


class FakeMLClient:
    def __init__(self, pages):
        self.jobs = FakeJobsList(pages)


@pytest.fixture
def stub_client(monkeypatch):
    def _install(pages):
        monkeypatch.setattr(jobcontrol, "get_ml_client", lambda: FakeMLClient(pages))
    return _install


def test_list_jobs_prints_header_and_rows(stub_client, capsys):
    pages = [[
        FakeJob("jolly_card_p6yh0phzxm", "jolly_card_p6yh", "login-5n2kkmvhk", "Completed"),
        FakeJob("cool_pig_mwhdcjs72n", "localtest-f4s", "f4s", "Failed"),
    ]]
    stub_client(pages)
    list_jobs("qstat")
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("JOBID")
    assert "NAME" in lines[0] and "PARTITION" in lines[0] and "STATE" in lines[0]
    # both jobs and their states are present
    body = "\n".join(lines[1:])
    assert "jolly_card_p6yh0phzxm" in body and "Completed" in body
    assert "cool_pig_mwhdcjs72n" in body and "Failed" in body


def test_list_jobs_handles_multiple_pages(stub_client, capsys):
    pages = [
        [FakeJob("a", "da", "f16s", "Running")],
        [FakeJob("b", "db", "hbv2", "Completed")],
    ]
    stub_client(pages)
    list_jobs("bjobs")
    body = capsys.readouterr().out
    assert "Running" in body and "Completed" in body


def test_list_jobs_tolerates_missing_fields(stub_client, capsys):
    # None fields must not blow up the fixed-width formatting
    pages = [[FakeJob(None, None, None, None)]]
    stub_client(pages)
    list_jobs("qstat")  # should not raise
    out = capsys.readouterr().out
    assert out  # header at least


def test_qstat_delegates_to_list_jobs(monkeypatch):
    seen = {}
    monkeypatch.setattr("amlhpc.jobcontrol.list_jobs", lambda prog: seen.setdefault("prog", prog))
    qstat([])
    assert seen["prog"] == "qstat"


def test_bjobs_delegates_to_list_jobs(monkeypatch):
    seen = {}
    monkeypatch.setattr("amlhpc.jobcontrol.list_jobs", lambda prog: seen.setdefault("prog", prog))
    bjobs([])
    assert seen["prog"] == "bjobs"


def test_qdel_delegates_to_cancel_job(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "amlhpc.jobcontrol.cancel_job",
        lambda prog, vargs: seen.update(prog=prog, vargs=vargs),
    )
    qdel(["job1", "job2"])
    assert seen == {"prog": "qdel", "vargs": ["job1", "job2"]}


def test_bkill_delegates_to_cancel_job(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "amlhpc.jobcontrol.cancel_job",
        lambda prog, vargs: seen.update(prog=prog, vargs=vargs),
    )
    bkill(["jobX"])
    assert seen == {"prog": "bkill", "vargs": ["jobX"]}
