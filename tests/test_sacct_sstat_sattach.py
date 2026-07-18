"""Tests for the sacct/sstat/sattach job-introspection commands.

These back the Slurm ``sacct`` (single-job status), ``sstat`` (node CPU/memory
utilization via Azure Monitor) and ``sattach`` (log tail/follow) commands. All
sit in ``amlhpc/jobcontrol.py`` and reuse ``get_ml_client``; the thin
``slurm/*.py`` wrappers just delegate. Azure access is fully stubbed.
"""

import pytest
from azure.core.exceptions import ResourceNotFoundError

import amlhpc.jobcontrol as jobcontrol
from amlhpc.jobcontrol import show_job_status, show_job_stats, attach_job


class FakeJob:
    def __init__(self, name, display_name="d", compute="f4s", status="Completed",
                 properties=None):
        self.name = name
        self.display_name = display_name
        self.compute = compute
        self.status = status
        self.properties = properties or {}
        self.creation_context = None


class FakeJobs:
    def __init__(self, jobs=None, missing=(), stream_calls=None, downloads=None):
        self._jobs = {j.name: j for j in (jobs or [])}
        self._missing = set(missing)
        self.stream_calls = stream_calls if stream_calls is not None else []
        self._downloads = downloads

    def get(self, jobid):
        if jobid in self._missing:
            raise ResourceNotFoundError("not found")
        if jobid in self._jobs:
            return self._jobs[jobid]
        raise ResourceNotFoundError("not found")

    def stream(self, jobid):
        self.stream_calls.append(jobid)

    def download(self, jobid, download_path=None):
        if self._downloads is None:
            return
        self._downloads(jobid, download_path)


class FakeMLClient:
    def __init__(self, **kw):
        self.jobs = FakeJobs(**kw)


@pytest.fixture
def stub_client(monkeypatch):
    def _install(**kw):
        client = FakeMLClient(**kw)
        monkeypatch.setattr(jobcontrol, "get_ml_client", lambda: client)
        return client
    return _install


def test_sacct_prints_header_and_status(stub_client, capsys):
    stub_client(jobs=[FakeJob("job_abc", "myjob", "f16s", "Running",
                              properties={"StartTimeUtc": "2026-07-18T11:50:00Z",
                                          "EndTimeUtc": "2026-07-18T12:36:00Z"})])
    show_job_status("sacct", ["job_abc"])
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].startswith("JOBID")
    assert "STATE" in lines[0] and "START" in lines[0] and "END" in lines[0]
    body = "\n".join(lines[1:])
    assert "job_abc" in body and "Running" in body
    assert "2026-07-18T11:50:00" in body and "2026-07-18T12:36:00" in body


def test_sacct_missing_job_exits_nonzero(stub_client, capsys):
    stub_client(missing={"ghost"})
    with pytest.raises(SystemExit) as exc:
        show_job_status("sacct", ["ghost"])
    assert exc.value.code == 1
    assert "sacct: job 'ghost' not found" in capsys.readouterr().out


def test_sacct_requires_jobid(stub_client):
    stub_client()
    with pytest.raises(SystemExit) as exc:
        show_job_status("sacct", [])
    assert exc.value.code != 0


def test_sattach_follow_calls_stream(stub_client):
    client = stub_client(jobs=[FakeJob("job_run", status="Running")])
    attach_job("sattach", ["job_run", "-f"])
    assert client.jobs.stream_calls == ["job_run"]


def test_sattach_missing_job_exits_nonzero(stub_client, capsys):
    stub_client(missing={"ghost"})
    with pytest.raises(SystemExit) as exc:
        attach_job("sattach", ["ghost"])
    assert exc.value.code == 1
    assert "sattach: job 'ghost' not found" in capsys.readouterr().out


def test_sattach_oneshot_prints_std_log(stub_client, capsys):
    import os

    def _download(jobid, download_path):
        d = os.path.join(download_path, "artifacts", "user_logs")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "std_log.txt"), "w") as fh:
            fh.write("hello from the job\n")

    stub_client(jobs=[FakeJob("job_done")], downloads=_download)
    attach_job("sattach", ["job_done"])
    assert "hello from the job" in capsys.readouterr().out


def test_sattach_oneshot_no_log_exits_nonzero(stub_client, capsys):
    def _download(jobid, download_path):
        pass

    stub_client(jobs=[FakeJob("job_empty")], downloads=_download)
    with pytest.raises(SystemExit) as exc:
        attach_job("sattach", ["job_empty"])
    assert exc.value.code == 1
    assert "no std_log.txt found" in capsys.readouterr().out


def test_sstat_missing_job_exits_nonzero(stub_client, capsys):
    stub_client(missing={"ghost"})
    with pytest.raises(SystemExit) as exc:
        show_job_stats("sstat", ["ghost"])
    assert exc.value.code == 1
    assert "sstat: job 'ghost' not found" in capsys.readouterr().out


def test_sstat_no_start_time_exits_nonzero(stub_client, capsys):
    stub_client(jobs=[FakeJob("job_queued", status="Queued", properties={})])
    with pytest.raises(SystemExit) as exc:
        show_job_stats("sstat", ["job_queued"])
    assert exc.value.code == 1
    assert "no start time" in capsys.readouterr().out


def test_sstat_queries_azure_monitor(stub_client, capsys, monkeypatch):
    monkeypatch.setenv("SUBSCRIPTION", "sub")
    monkeypatch.setenv("CI_RESOURCE_GROUP", "rg")
    monkeypatch.setenv("CI_WORKSPACE", "ws")
    stub_client(jobs=[FakeJob("job_abc", status="Completed",
                              properties={"StartTimeUtc": "2026-07-18T11:50:00Z",
                                          "EndTimeUtc": "2026-07-18T12:36:00Z"})])

    class FakeDatum:
        def __init__(self, ts, avg, mx):
            self.timestamp = ts
            self.average = avg
            self.maximum = mx

    class FakeTS:
        def __init__(self, data):
            self.data = data

    class FakeMetric:
        def __init__(self, name, timeseries):
            self.name = name
            self.timeseries = timeseries

    class FakeResponse:
        def __init__(self):
            self.metrics = [
                FakeMetric("CpuUtilizationPercentage",
                           [FakeTS([FakeDatum("2026-07-18T12:00:00", 4.0, 26.0),
                                    FakeDatum("2026-07-18T12:36:00", 80.0, 84.0)])]),
                FakeMetric("CpuMemoryUtilizationPercentage", [FakeTS([])]),
            ]

    class FakeMetricsClient:
        def __init__(self, cred):
            pass

        def query_resource(self, uri, **kw):
            assert "Microsoft.MachineLearningServices/workspaces/ws" in uri
            return FakeResponse()

    import azure.monitor.query as amq
    monkeypatch.setattr(amq, "MetricsQueryClient", FakeMetricsClient)
    show_job_stats("sstat", ["job_abc"])
    out = capsys.readouterr().out
    assert "CpuUtilizationPercentage" in out
    assert "84.0%" in out
    assert "no data" in out


def test_wrappers_delegate(monkeypatch):
    from amlhpc.slurm.sacct import sacct
    from amlhpc.slurm.sstat import sstat
    from amlhpc.slurm.sattach import sattach

    seen = {}
    monkeypatch.setattr("amlhpc.jobcontrol.show_job_status",
                        lambda prog, vargs: seen.update(sacct=(prog, vargs)))
    monkeypatch.setattr("amlhpc.jobcontrol.show_job_stats",
                        lambda prog, vargs: seen.update(sstat=(prog, vargs)))
    monkeypatch.setattr("amlhpc.jobcontrol.attach_job",
                        lambda prog, vargs: seen.update(sattach=(prog, vargs)))
    sacct(["j1"])
    sstat(["j2"])
    sattach(["j3", "-f"])
    assert seen["sacct"] == ("sacct", ["j1"])
    assert seen["sstat"] == ("sstat", ["j2"])
    assert seen["sattach"] == ("sattach", ["j3", "-f"])
