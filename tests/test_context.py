"""Tests for the connection profile store and resolver (amlhpc/context.py).

``resolve_connection`` decides which Azure Machine Learning workspace every
command talks to, through a documented precedence chain. These tests isolate
``$HOME`` (so the real ~/.amlhpc is never touched) and the AML env vars (scrubbed
by conftest) to exercise each rung of that chain plus the profile read/write and
share/import round-trip. No Azure access is involved -- the module is pure.
"""

import json

import pytest

import amlhpc.context as context
from amlhpc.context import (
    ConnectionNotConfigured,
    export_profile,
    import_profile,
    put_profile,
    resolve_connection,
    use_profile,
)


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    """Point ~ at a temp dir so config lands in an isolated ~/.amlhpc."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


def test_put_profile_writes_and_marks_current():
    # Given a fresh config, When a profile is added as current,
    put_profile("prod", "sub-1", "rg-1", "ws-1")
    # Then resolving with no hints returns it, sourced from the profile.
    conn = resolve_connection()
    assert conn.subscription == "sub-1"
    assert conn.resource_group == "rg-1"
    assert conn.workspace == "ws-1"
    assert conn.source == "profile:prod"


def test_config_file_is_0600(_home):
    # Given a written profile, Then the config file is owner-only (identity data).
    put_profile("prod", "sub-1", "rg-1", "ws-1")
    path = _home / ".amlhpc" / "config.json"
    mode = path.stat().st_mode & 0o777
    assert mode == 0o600


def test_flag_beats_env_and_current(monkeypatch):
    # Given two profiles and a legacy env pointing elsewhere,
    put_profile("prod", "sub-prod", "rg-prod", "ws-prod")
    put_profile("dev", "sub-dev", "rg-dev", "ws-dev", make_current=True)
    monkeypatch.setenv("SUBSCRIPTION", "sub-env")
    monkeypatch.setenv("CI_RESOURCE_GROUP", "rg-env")
    monkeypatch.setenv("CI_WORKSPACE", "ws-env")
    # When --cluster names prod, Then prod wins over env and current.
    conn = resolve_connection(cluster="prod")
    assert conn.workspace == "ws-prod"
    assert conn.source == "flag:prod"


def test_amlhpc_cluster_env_beats_legacy_env(monkeypatch):
    # Given a profile and AMLHPC_CLUSTER naming it, plus legacy env vars,
    put_profile("prod", "sub-prod", "rg-prod", "ws-prod", make_current=False)
    monkeypatch.setenv("AMLHPC_CLUSTER", "prod")
    monkeypatch.setenv("SUBSCRIPTION", "sub-env")
    monkeypatch.setenv("CI_RESOURCE_GROUP", "rg-env")
    monkeypatch.setenv("CI_WORKSPACE", "ws-env")
    # When resolving, Then AMLHPC_CLUSTER selects the profile over legacy env.
    conn = resolve_connection()
    assert conn.workspace == "ws-prod"
    assert conn.source == "env:AMLHPC_CLUSTER"


def test_legacy_env_beats_current_profile(monkeypatch):
    # Given a current profile AND all three legacy env vars set,
    put_profile("prod", "sub-prod", "rg-prod", "ws-prod", make_current=True)
    monkeypatch.setenv("SUBSCRIPTION", "sub-env")
    monkeypatch.setenv("CI_RESOURCE_GROUP", "rg-env")
    monkeypatch.setenv("CI_WORKSPACE", "ws-env")
    # When resolving with no flag/AMLHPC_CLUSTER, Then legacy env wins
    # (backward compatibility: existing shells and in-AML CIs keep working).
    conn = resolve_connection()
    assert conn.workspace == "ws-env"
    assert conn.source == "env:legacy"


def test_partial_legacy_env_falls_through_to_current(monkeypatch):
    # Given only two of the three legacy vars (incomplete) and a current profile,
    put_profile("prod", "sub-prod", "rg-prod", "ws-prod", make_current=True)
    monkeypatch.setenv("SUBSCRIPTION", "sub-env")
    monkeypatch.setenv("CI_RESOURCE_GROUP", "rg-env")
    # When resolving, Then the incomplete env is ignored and current wins.
    conn = resolve_connection()
    assert conn.workspace == "ws-prod"
    assert conn.source == "profile:prod"


def test_nothing_configured_raises_with_guidance():
    # Given no profiles and no env, When resolving, Then a guiding error is raised.
    with pytest.raises(ConnectionNotConfigured) as excinfo:
        resolve_connection()
    assert "deploy connect" in str(excinfo.value)


def test_unknown_cluster_flag_lists_known():
    # Given one profile, When --cluster names a missing one,
    put_profile("prod", "sub-1", "rg-1", "ws-1")
    # Then the error names the missing profile and the known ones.
    with pytest.raises(ConnectionNotConfigured) as excinfo:
        resolve_connection(cluster="ghost")
    message = str(excinfo.value)
    assert "ghost" in message
    assert "prod" in message


def test_use_profile_switches_current():
    # Given two profiles, When use_profile picks the non-current one,
    put_profile("prod", "sub-prod", "rg-prod", "ws-prod", make_current=True)
    put_profile("dev", "sub-dev", "rg-dev", "ws-dev", make_current=False)
    use_profile("dev")
    # Then resolving returns dev.
    conn = resolve_connection()
    assert conn.workspace == "ws-dev"


def test_use_unknown_profile_raises():
    # Given no such profile, When use_profile is called, Then it raises.
    with pytest.raises(ConnectionNotConfigured):
        use_profile("ghost")


def test_workspace_uri_is_arm_id():
    # Given a resolved connection, Then workspace_uri builds the ARM resource ID.
    put_profile("prod", "sub-1", "rg-1", "ws-1")
    conn = resolve_connection()
    assert conn.workspace_uri() == (
        "/subscriptions/sub-1/resourceGroups/rg-1/providers/"
        "Microsoft.MachineLearningServices/workspaces/ws-1")


def test_export_profile_is_secret_free_and_scoped():
    # Given two profiles, When one is exported,
    put_profile("prod", "sub-prod", "rg-prod", "ws-prod")
    put_profile("dev", "sub-dev", "rg-dev", "ws-dev")
    blob = export_profile("prod")
    # Then only that profile's identifiers are present -- no current, no other
    # clusters, no credential fields.
    assert blob == {
        "amlhpc_profile": "prod",
        "subscription": "sub-prod",
        "resource_group": "rg-prod",
        "workspace": "ws-prod",
    }


def test_share_import_round_trip():
    # Given a profile exported on "machine A",
    put_profile("prod", "sub-prod", "rg-prod", "ws-prod")
    blob = export_profile("prod")
    # When a JSON round-trip (the wire) and import happens on "machine B",
    wire = json.loads(json.dumps(blob))
    name = import_profile(wire)
    # Then the profile is registered and resolvable by name.
    assert name == "prod"
    conn = resolve_connection(cluster="prod")
    assert conn.workspace == "ws-prod"


def test_import_malformed_blob_raises():
    # Given a blob missing the workspace field, When imported, Then it is rejected.
    with pytest.raises(ConnectionNotConfigured):
        import_profile({"amlhpc_profile": "x", "subscription": "s",
                        "resource_group": "r"})
