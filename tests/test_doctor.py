"""Tests for 'deploy doctor' feature-completeness checks (amlhpc/deploy.py).

The checklist logic is pure: run_doctor_checks(ml_client) probes a workspace
and returns structured records without printing or touching Azure. These tests
drive it with a stub ml_client whose environments / compute / datastores are
tunable, so every check (present and missing) is exercised offline, plus the
--fix path that creates the default environment.
"""

import pytest

import amlhpc.deploy as deploy
from amlhpc.deploy import (
    DEFAULT_ENVIRONMENT,
    create_default_environment,
    run_doctor_checks,
)


class _Named:
    def __init__(self, name, kind=None):
        self.name = name
        if kind is not None:
            self.type = kind


class _Environments:
    def __init__(self, names):
        self._names = list(names)
        self.created = []

    def list(self, name=None):
        for n in self._names:
            if name is None or n == name:
                yield _Named(n)

    def create_or_update(self, environment):
        self._names.append(environment.name)
        self.created.append(environment.name)
        return _Named(environment.name)


class _Compute:
    def __init__(self, items):
        self._items = list(items)

    def list(self):
        return list(self._items)


class _Datastores:
    def __init__(self, names):
        self._names = set(names)

    def get(self, name):
        if name not in self._names:
            raise KeyError(name)
        return _Named(name)


class _MLClient:
    def __init__(self, environments=(), compute=(), datastores=()):
        self.environments = _Environments(environments)
        self.compute = _Compute(compute)
        self.datastores = _Datastores(datastores)


def _by_name(checks, prefix):
    for check in checks:
        if check["name"].startswith(prefix):
            return check
    raise AssertionError("no check starting with " + prefix)


def _healthy_client():
    return _MLClient(
        environments=[DEFAULT_ENVIRONMENT],
        compute=[_Named("f4s", "amlcompute"), _Named("login", "computeinstance")],
        datastores=["workspaceblobstore"],
    )


def test_all_checks_pass_on_healthy_workspace():
    checks = run_doctor_checks(_healthy_client())
    assert all(c["ok"] for c in checks)


def test_missing_environment_is_flagged_and_fixable():
    client = _MLClient(
        environments=[],
        compute=[_Named("f4s", "amlcompute"), _Named("login", "computeinstance")],
        datastores=["workspaceblobstore"],
    )
    env_check = _by_name(run_doctor_checks(client), "default job environment")
    assert env_check["ok"] is False
    assert env_check["fixable"] is True


def test_missing_partition_is_flagged_not_fixable():
    client = _MLClient(
        environments=[DEFAULT_ENVIRONMENT],
        compute=[_Named("login", "computeinstance")],
        datastores=["workspaceblobstore"],
    )
    part_check = _by_name(run_doctor_checks(client), "at least one AmlCompute")
    assert part_check["ok"] is False
    assert part_check["fixable"] is False


def test_missing_login_instance_is_flagged():
    client = _MLClient(
        environments=[DEFAULT_ENVIRONMENT],
        compute=[_Named("f4s", "amlcompute")],
        datastores=["workspaceblobstore"],
    )
    ci_check = _by_name(run_doctor_checks(client), "login ComputeInstance")
    assert ci_check["ok"] is False


def test_missing_datastore_is_flagged():
    client = _MLClient(
        environments=[DEFAULT_ENVIRONMENT],
        compute=[_Named("f4s", "amlcompute"), _Named("login", "computeinstance")],
        datastores=[],
    )
    ds_check = _by_name(run_doctor_checks(client), "default datastore")
    assert ds_check["ok"] is False


def test_create_default_environment_makes_it_pass():
    client = _MLClient(
        environments=[],
        compute=[_Named("f4s", "amlcompute"), _Named("login", "computeinstance")],
        datastores=["workspaceblobstore"],
    )
    name = create_default_environment(client)
    assert name == DEFAULT_ENVIRONMENT
    # After creation the environment check now passes.
    env_check = _by_name(run_doctor_checks(client), "default job environment")
    assert env_check["ok"] is True


def test_deploy_doctor_fix_creates_env_and_reports_ready(monkeypatch, capsys):
    client = _MLClient(
        environments=[],
        compute=[_Named("f4s", "amlcompute"), _Named("login", "computeinstance")],
        datastores=["workspaceblobstore"],
    )
    monkeypatch.setattr(deploy, "config_ml_client", lambda cluster=None: client)

    class Args:
        fix = True

    deploy.deploy_doctor(Args())
    out = capsys.readouterr().out
    assert "created default environment: " + DEFAULT_ENVIRONMENT in out
    assert "amlhpc-ready" in out


def test_deploy_doctor_report_only_exits_nonzero_when_incomplete(monkeypatch, capsys):
    client = _MLClient(
        environments=[],
        compute=[_Named("f4s", "amlcompute"), _Named("login", "computeinstance")],
        datastores=["workspaceblobstore"],
    )
    monkeypatch.setattr(deploy, "config_ml_client", lambda cluster=None: client)

    class Args:
        fix = False

    with pytest.raises(SystemExit) as excinfo:
        deploy.deploy_doctor(Args())
    assert excinfo.value.code == 1
    assert "--fix" in capsys.readouterr().out


class _ConnectArgs:
    def __init__(self, name, resource_group, workspace, subscription=None,
                 no_current=False, check=False):
        self.name = name
        self.resource_group = resource_group
        self.workspace = workspace
        self.subscription = subscription
        self.no_current = no_current
        self.check = check


def test_deploy_connect_registers_profile_from_env_subscription(
        tmp_path, monkeypatch, capsys):
    # Given an isolated HOME and $SUBSCRIPTION set,
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("SUBSCRIPTION", "sub-env")
    # When connecting without -s, Then the env subscription is used and the
    # profile resolves as the current cluster.
    deploy.deploy_connect(_ConnectArgs("prod", "rg-1", "ws-1"))
    from amlhpc.context import resolve_connection
    conn = resolve_connection()
    assert (conn.subscription, conn.resource_group, conn.workspace) == (
        "sub-env", "rg-1", "ws-1")
    assert "registered cluster profile 'prod'" in capsys.readouterr().out


def test_deploy_connect_explicit_subscription_and_no_current(
        tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("SUBSCRIPTION", raising=False)
    # Given --no-current, the profile is stored but not made current,
    deploy.deploy_connect(_ConnectArgs(
        "dev", "rg-d", "ws-d", subscription="sub-explicit", no_current=True))
    from amlhpc.context import load_config, resolve_connection
    config = load_config()
    assert config["current"] is None
    assert config["clusters"]["dev"]["subscription"] == "sub-explicit"
    # but is still reachable by explicit --cluster.
    conn = resolve_connection(cluster="dev")
    assert conn.workspace == "ws-d"


def test_deploy_connect_missing_subscription_exits(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("SUBSCRIPTION", raising=False)
    with pytest.raises(SystemExit):
        deploy.deploy_connect(_ConnectArgs("x", "rg", "ws"))


class _ShareArgs:
    def __init__(self, name, output=None):
        self.name = name
        self.output = output


class _ImportArgs:
    def __init__(self, file=None, no_current=False, check=False):
        self.file = file
        self.no_current = no_current
        self.check = check


def test_share_to_file_then_import_round_trips(tmp_path, monkeypatch):
    # Given a profile registered on "machine A",
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    from amlhpc.context import put_profile, resolve_connection
    put_profile("prod", "sub-1", "rg-1", "ws-1")
    pointer = tmp_path / "prod.json"
    deploy.deploy_share(_ShareArgs("prod", output=str(pointer)))

    # When a second machine (fresh HOME) imports the pointer file,
    home_b = tmp_path / "userB"
    home_b.mkdir()
    monkeypatch.setenv("HOME", str(home_b))
    monkeypatch.setenv("USERPROFILE", str(home_b))
    deploy.deploy_import(_ImportArgs(file=str(pointer)))

    # Then the profile resolves identically on machine B.
    conn = resolve_connection(cluster="prod")
    assert (conn.subscription, conn.resource_group, conn.workspace) == (
        "sub-1", "rg-1", "ws-1")


def test_share_pointer_carries_no_secrets_or_extra_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    from amlhpc.context import put_profile
    put_profile("prod", "sub-1", "rg-1", "ws-1")
    put_profile("dev", "sub-2", "rg-2", "ws-2")
    pointer = tmp_path / "prod.json"
    deploy.deploy_share(_ShareArgs("prod", output=str(pointer)))
    import json
    blob = json.loads(pointer.read_text())
    # Only the one profile's four identifier fields -- no current pointer, no
    # other clusters, no credential material.
    assert set(blob) == {"amlhpc_profile", "subscription", "resource_group",
                         "workspace"}


def test_import_malformed_pointer_exits(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    bad = tmp_path / "bad.json"
    bad.write_text("not json at all")
    with pytest.raises(SystemExit):
        deploy.deploy_import(_ImportArgs(file=str(bad)))


class _InviteArgs:
    def __init__(self, user, cluster=None, role="AzureML Data Scientist", yes=True):
        self.user = user
        self.cluster = cluster
        self.role = role
        self.yes = yes


class _FakeRun:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


def test_build_role_command_is_workspace_scoped():
    scope = ("/subscriptions/sub-1/resourceGroups/rg-1/providers/"
             "Microsoft.MachineLearningServices/workspaces/ws-1")
    cmd = deploy.build_role_command(
        "create", "/usr/bin/az", "AzureML Data Scientist", scope, "user@example.com")
    assert cmd == ["/usr/bin/az", "role", "assignment", "create",
                   "--role", "AzureML Data Scientist",
                   "--scope", scope,
                   "--assignee", "user@example.com"]


def _profile(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    from amlhpc.context import put_profile
    put_profile("prod", "sub-1", "rg-1", "ws-1")


def test_invite_without_az_prints_fallback_command(tmp_path, monkeypatch, capsys):
    _profile(tmp_path, monkeypatch)
    # Given no az on PATH, When inviting, Then the exact command is printed for
    # an admin -- and nothing is executed.
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: None)
    deploy.deploy_invite(_InviteArgs("user@example.com"))
    out = capsys.readouterr().out
    assert "az" in out and "role assignment create" in out
    assert "workspaces/ws-1" in out


def test_invite_runs_az_on_confirmation(tmp_path, monkeypatch, capsys):
    _profile(tmp_path, monkeypatch)
    import shutil
    import subprocess
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/az")
    calls = {}

    def fake_run(command, capture_output=False, text=False):
        calls["command"] = command
        return _FakeRun(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    deploy.deploy_invite(_InviteArgs("user@example.com", yes=True))
    out = capsys.readouterr().out
    assert calls["command"][3] == "create"
    assert "--assignee" in calls["command"]
    assert "granted access for 'user@example.com'" in out


def test_invite_abort_on_declined_confirmation(tmp_path, monkeypatch, capsys):
    _profile(tmp_path, monkeypatch)
    import shutil
    import subprocess
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/az")
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    ran = {"called": False}

    def fake_run(command, capture_output=False, text=False):
        ran["called"] = True
        return _FakeRun(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    deploy.deploy_invite(_InviteArgs("user@example.com", yes=False))
    assert ran["called"] is False
    assert "aborted" in capsys.readouterr().out


def test_uninvite_failure_prints_admin_fallback_and_exits(tmp_path, monkeypatch, capsys):
    _profile(tmp_path, monkeypatch)
    import shutil
    import subprocess
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/az")

    def fake_run(command, capture_output=False, text=False):
        return _FakeRun(returncode=1, stderr="AuthorizationFailed")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(SystemExit):
        deploy.deploy_uninvite(_InviteArgs("user@example.com", yes=True))
    out = capsys.readouterr().out
    assert "role assignment delete" in out
    assert "ask an admin" in out
