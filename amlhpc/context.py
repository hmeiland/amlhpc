"""Cluster connection profiles: a portable, OS-neutral replacement for the
loose ``SUBSCRIPTION`` / ``CI_RESOURCE_GROUP`` / ``CI_WORKSPACE`` exports.

A *connection* is just three identifiers that name an Azure Machine Learning
workspace (subscription, resource group, workspace). Historically the client
read them from three environment variables, which are per-shell and per-OS:
a Linux user ``source``s a ``.sh`` file, a Windows user cannot, and switching
between clusters means re-exporting all three. None of the three are secret --
they confer no access on their own (authentication is always the caller's own
Azure identity via ``DefaultAzureCredential``), so they can be stored as data.

This module keeps them as data in ``~/.amlhpc/config.json`` -- read directly by
the tool on any OS, so the same file works identically from bash, PowerShell,
cmd or inside AML. ``resolve_connection`` applies a precedence chain so nothing
that works today breaks:

  1. an explicit ``--cluster NAME`` on the command line,
  2. the ``AMLHPC_CLUSTER`` environment variable (names a stored profile),
  3. the legacy ``SUBSCRIPTION`` / ``CI_RESOURCE_GROUP`` / ``CI_WORKSPACE``
     environment variables (backward compatible -- the in-AML ComputeInstance
     sets CI_RESOURCE_GROUP / CI_WORKSPACE automatically),
  4. the ``current`` profile recorded in ``~/.amlhpc/config.json``.

The file is stored with ``0600`` permissions because it records subscription /
resource-group identity; it never holds credentials or tokens.
"""

import json
import os


class ConnectionNotConfigured(Exception):
    """No usable connection could be resolved from any source.

    Carries the human-facing guidance so the CLI boundary can print it and
    exit non-zero without re-deriving the message.
    """

    def __init__(self, message):
        super().__init__(message)
        self.message = message


class Connection:
    """A resolved (subscription, resource_group, workspace) triple.

    Immutable value object: the fields are set once at construction and the
    class exposes no mutators, so a resolved connection cannot drift.
    """

    __slots__ = ("subscription", "resource_group", "workspace", "source")

    def __init__(self, subscription, resource_group, workspace, source):
        self.subscription = subscription
        self.resource_group = resource_group
        self.workspace = workspace
        self.source = source

    def workspace_uri(self):
        """Return the ARM resource ID of the workspace (used by sstat)."""
        return (
            "/subscriptions/" + self.subscription
            + "/resourceGroups/" + self.resource_group
            + "/providers/Microsoft.MachineLearningServices/workspaces/"
            + self.workspace
        )


def config_path():
    """Return the path to ~/.amlhpc/config.json, honouring $HOME on every OS."""
    from pathlib import Path

    return str(Path.home() / ".amlhpc" / "config.json")


def load_config():
    """Return the parsed config dict, or an empty skeleton if none exists.

    A missing or unreadable file yields the empty skeleton rather than an
    error: an unconfigured client still works through the legacy env vars.
    """
    path = config_path()
    if not os.path.isfile(path):
        return {"current": None, "clusters": {}}
    with open(path, "r") as handle:
        data = json.load(handle)
    data.setdefault("current", None)
    data.setdefault("clusters", {})
    return data


def save_config(config):
    """Write the config dict to ~/.amlhpc/config.json with 0600 permissions."""
    path = config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(path, 0o600)
    return path


def put_profile(name, subscription, resource_group, workspace, make_current=True):
    """Add or replace a named cluster profile; optionally mark it current."""
    config = load_config()
    config["clusters"][name] = {
        "subscription": subscription,
        "resource_group": resource_group,
        "workspace": workspace,
    }
    if make_current:
        config["current"] = name
    return save_config(config)


def use_profile(name):
    """Set the current profile to an existing named profile.

    Returns the profile dict. Raises ConnectionNotConfigured naming the available
    profiles when the name is unknown, so the caller need not re-list them.
    """
    config = load_config()
    if name not in config["clusters"]:
        known = ", ".join(sorted(config["clusters"])) or "(none)"
        raise ConnectionNotConfigured(
            "no cluster profile named '" + name + "'; known profiles: " + known)
    config["current"] = name
    save_config(config)
    return config["clusters"][name]


def _from_profile(profile, source):
    """Build a Connection from a stored profile dict, or raise if incomplete."""
    try:
        return Connection(
            subscription=profile["subscription"],
            resource_group=profile["resource_group"],
            workspace=profile["workspace"],
            source=source,
        )
    except KeyError as missing:
        raise ConnectionNotConfigured(
            "cluster profile is missing field " + str(missing)) from None


def resolve_connection(cluster=None):
    """Resolve the active connection through the precedence chain.

    ``cluster`` is the optional ``--cluster NAME`` flag value (highest
    precedence). Raises ConnectionNotConfigured with actionable guidance when nothing
    resolves, so every CLI command shares one clear failure message.
    """
    config = load_config()

    if cluster is not None:
        if cluster not in config["clusters"]:
            known = ", ".join(sorted(config["clusters"])) or "(none)"
            raise ConnectionNotConfigured(
                "no cluster profile named '" + cluster + "'; known profiles: "
                + known + " (add one with 'deploy connect' or 'deploy init')")
        return _from_profile(config["clusters"][cluster], "flag:" + cluster)

    env_cluster = os.environ.get("AMLHPC_CLUSTER")
    if env_cluster:
        if env_cluster not in config["clusters"]:
            known = ", ".join(sorted(config["clusters"])) or "(none)"
            raise ConnectionNotConfigured(
                "AMLHPC_CLUSTER names '" + env_cluster + "' but no such profile "
                "exists; known profiles: " + known)
        return _from_profile(config["clusters"][env_cluster], "env:AMLHPC_CLUSTER")

    subscription = os.environ.get("SUBSCRIPTION")
    resource_group = os.environ.get("CI_RESOURCE_GROUP")
    workspace = os.environ.get("CI_WORKSPACE")
    if subscription and resource_group and workspace:
        return Connection(subscription, resource_group, workspace, "env:legacy")

    current = config["current"]
    if current and current in config["clusters"]:
        return _from_profile(config["clusters"][current], "profile:" + current)

    raise ConnectionNotConfigured(
        "no Azure Machine Learning connection configured. Set one with "
        "'deploy connect -g <rg> -w <workspace>', 'deploy init', or by "
        "exporting SUBSCRIPTION / CI_RESOURCE_GROUP / CI_WORKSPACE.")


def export_profile(name):
    """Return a portable, secret-free dict describing one profile.

    This is what 'deploy share' serialises. It deliberately carries only the
    three identifiers plus the name -- no credentials, no current-pointer, no
    other clusters -- so it is safe to send over any channel.
    """
    config = load_config()
    if name not in config["clusters"]:
        known = ", ".join(sorted(config["clusters"])) or "(none)"
        raise ConnectionNotConfigured(
            "no cluster profile named '" + name + "'; known profiles: " + known)
    profile = config["clusters"][name]
    return {
        "amlhpc_profile": name,
        "subscription": profile["subscription"],
        "resource_group": profile["resource_group"],
        "workspace": profile["workspace"],
    }


def import_profile(blob, make_current=True):
    """Merge a shared profile (dict from export_profile) into the local config.

    Returns the imported profile's name. Raises ConnectionNotConfigured if the blob is
    missing a required field, so a malformed share is rejected at the boundary.
    """
    try:
        name = blob["amlhpc_profile"]
        subscription = blob["subscription"]
        resource_group = blob["resource_group"]
        workspace = blob["workspace"]
    except KeyError as missing:
        raise ConnectionNotConfigured(
            "shared profile is missing field " + str(missing)) from None
    put_profile(name, subscription, resource_group, workspace,
                make_current=make_current)
    return name
