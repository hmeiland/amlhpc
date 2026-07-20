"""Tests for 'deploy partition' AmlCompute creation (amlhpc/deploy.py).

A partition created without a managed identity gets identity: null and then
cannot pull images from the workspace-attached ACR, so deploy_partition must
stamp a SystemAssigned identity on the AmlCompute it submits. Azure is stubbed:
resolve_connection, the MLClient and the credential are all replaced, and the
AmlCompute passed to begin_create_or_update is captured and inspected.
"""

import types

import amlhpc.deploy as deploy


class _Poller:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class _Compute:
    def __init__(self):
        self.created = None

    def begin_create_or_update(self, compute):
        self.created = compute
        returned = types.SimpleNamespace(name=compute.name, size=compute.size,
                                         max_instances=compute.max_instances)
        return _Poller(returned)


class _FakeMLClient:
    last = None

    def __init__(self, **kw):
        self.compute = _Compute()
        _FakeMLClient.last = self


def _args(**over):
    base = dict(name="hbv3", size="Standard_HB120rs_v3", min_nodes=0,
                max_nodes=4, idle_time=120, priority="Dedicated")
    base.update(over)
    return types.SimpleNamespace(**base)


def _stub_azure(monkeypatch):
    import azure.ai.ml
    import azure.identity

    monkeypatch.setattr(deploy, "resolve_connection",
                        lambda: types.SimpleNamespace(subscription="s", resource_group="rg", workspace="ws"),
                        raising=False)
    monkeypatch.setattr("amlhpc.context.resolve_connection",
                        lambda cluster=None: types.SimpleNamespace(subscription="s", resource_group="rg", workspace="ws"))
    monkeypatch.setattr(azure.ai.ml, "MLClient", _FakeMLClient)
    monkeypatch.setattr(azure.identity, "DefaultAzureCredential", lambda *a, **k: object())


def test_partition_gets_system_assigned_identity(monkeypatch, capsys):
    _stub_azure(monkeypatch)
    deploy.deploy_partition(_args())
    created = _FakeMLClient.last.compute.created
    assert created.identity is not None
    assert created.identity.type == "SystemAssigned"
    out = capsys.readouterr().out
    assert "hbv3" in out
