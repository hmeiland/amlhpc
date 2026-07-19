"""Tests for 'deploy validate' end-to-end feature exercise (amlhpc/deploy.py).

The costly parts (submitting real jobs, RBAC, Azure) are isolated behind pure
helpers and behind the same seams doctor uses: run_doctor_checks, the slurm
front-ends and config_ml_client are all monkeypatched, so the staged
orchestration is driven offline and every stage's recorded outcome is asserted,
plus the pure building blocks (state classifier, EESSI wrap, report, stdout
token capture) on their own.
"""

import pytest

import amlhpc.deploy as deploy
from amlhpc.deploy import (
    EESSI_INIT,
    ValidationReport,
    build_eessi_wrap,
    is_terminal_state,
)


def test_build_eessi_wrap_prefixes_mount_then_command():
    wrap = build_eessi_wrap("simpleFoam -help")
    assert wrap.startswith(EESSI_INIT + "; ")
    assert wrap.endswith("simpleFoam -help")


@pytest.mark.parametrize("status", ["Completed", "failed", "Canceled", "cancelled"])
def test_terminal_states_are_terminal(status):
    assert is_terminal_state(status) is True


@pytest.mark.parametrize("status", ["Running", "Queued", "Starting", None, ""])
def test_non_terminal_states_are_not_terminal(status):
    assert is_terminal_state(status) is False


def test_report_tracks_pass_fail_skip_and_all_ok():
    report = ValidationReport()
    report.ok("a")
    report.skip("b", "not requested")
    assert report.all_ok is True
    report.fail("c", "broke")
    assert report.all_ok is False
    assert [s["name"] for s in report.failed] == ["c"]


def test_capture_stdout_token_returns_last_token_and_echoes(capsys):
    def prints_jobid():
        print("gifted_engine_yq801rygm2")

    token = deploy._capture_stdout_token(prints_jobid)
    assert token == "gifted_engine_yq801rygm2"
    assert "gifted_engine_yq801rygm2" in capsys.readouterr().out


def test_capture_stdout_token_none_when_no_output():
    assert deploy._capture_stdout_token(lambda: None) is None


class _Named:
    def __init__(self, name, kind=None):
        self.name = name
        if kind is not None:
            self.type = kind


class _Job:
    def __init__(self, name, status="Completed"):
        self.name = name
        self.status = status


class _Page:
    def __init__(self, jobs):
        self._jobs = jobs

    def __iter__(self):
        return iter(self._jobs)


class _JobList:
    def __init__(self, jobs):
        self._jobs = jobs

    def by_page(self):
        return [_Page(self._jobs)]


class _Jobs:
    def __init__(self, jobs):
        self._jobs = {j.name: j for j in jobs}

    def list(self):
        return _JobList(list(self._jobs.values()))

    def get(self, name):
        return self._jobs[name]


class _Compute:
    def __init__(self, items):
        self._items = list(items)

    def list(self):
        return list(self._items)


class _MLClient:
    def __init__(self, compute, jobs):
        self.compute = _Compute(compute)
        self.jobs = _Jobs(jobs)


class _ValidateArgs:
    def __init__(self, **kw):
        self.cluster = kw.get("cluster")
        self.partition = kw.get("partition")
        self.include_container = kw.get("include_container", False)
        self.container_path = kw.get("container_path", "environments")
        self.include_dask = kw.get("include_dask", False)
        self.invite_user = kw.get("invite_user")
        self.timeout = kw.get("timeout", 1)
        self.keep = kw.get("keep", True)
        self.verbose = kw.get("verbose", 0)


def _wire_healthy(monkeypatch, tmp_path, jobid="probe_job_xyz"):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    from amlhpc.context import put_profile
    put_profile("prod", "sub-1", "rg-1", "ws-1")

    client = _MLClient(
        compute=[_Named("f4s", "amlcompute"), _Named("login", "computeinstance")],
        jobs=[_Job(jobid, "Completed"), _Job("srun_job", "Completed")],
    )
    monkeypatch.setattr(deploy, "config_ml_client", lambda cluster=None: client)
    monkeypatch.setattr(deploy, "run_doctor_checks",
                        lambda ml_client: [{"name": "env", "ok": True}])

    import amlhpc.slurm.sbatch as sbatch_mod
    import amlhpc.slurm.srun as srun_mod
    import amlhpc.jobcontrol as jobcontrol_mod
    monkeypatch.setattr(sbatch_mod, "sbatch", lambda argv: print(jobid))
    monkeypatch.setattr(srun_mod, "srun", lambda argv: print("srun_job"))
    monkeypatch.setattr(jobcontrol_mod, "attach_job", lambda prog, argv: print("log line"))
    monkeypatch.setattr(jobcontrol_mod, "show_job_stats", lambda prog, argv: print("util"))
    return client


def test_validate_healthy_workspace_all_core_stages_pass(tmp_path, monkeypatch, capsys):
    _wire_healthy(monkeypatch, tmp_path)
    deploy.deploy_validate(_ValidateArgs(keep=True))
    out = capsys.readouterr().out
    assert "workspace validated" in out
    for stage in ("connection", "doctor", "sinfo", "sbatch (EESSI)",
                  "squeue/qstat/bjobs", "sacct", "job watch", "sattach",
                  "sstat", "srun"):
        assert stage in out
    assert "0 failed" in out


def test_validate_no_partition_skips_job_stages_and_still_ok(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    from amlhpc.context import put_profile
    put_profile("prod", "sub-1", "rg-1", "ws-1")

    client = _MLClient(compute=[_Named("login", "computeinstance")], jobs=[])
    monkeypatch.setattr(deploy, "config_ml_client", lambda cluster=None: client)
    monkeypatch.setattr(deploy, "run_doctor_checks",
                        lambda ml_client: [{"name": "env", "ok": True}])

    deploy.deploy_validate(_ValidateArgs(keep=True))
    out = capsys.readouterr().out
    assert "no partition" in out


def test_validate_failed_job_marks_watch_failed_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    _wire_healthy(monkeypatch, tmp_path, jobid="probe_job_xyz")
    client = _MLClient(
        compute=[_Named("f4s", "amlcompute"), _Named("login", "computeinstance")],
        jobs=[_Job("probe_job_xyz", "Failed"), _Job("srun_job", "Completed")],
    )
    monkeypatch.setattr(deploy, "config_ml_client", lambda cluster=None: client)

    with pytest.raises(SystemExit) as excinfo:
        deploy.deploy_validate(_ValidateArgs(keep=True))
    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "validation FAILED" in out
    assert "job watch" in out
