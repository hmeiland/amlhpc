"""Site-wide job configuration stored in the workspace storage stack.

amlhpc keeps optional site-wide *prolog* and *epilog* shell snippets in the
workspace's default blob datastore (``workspaceblobstore``), under a well-known
prefix (default ``amlhpc/``). Because the snippets live in the storage stack
they are shared by every user and every job in the workspace: a workspace admin
uploads them once (``deploy config set-prolog ...``) and from then on every
``sbatch``/``srun`` job automatically runs the prolog before, and the epilog
after, the user command.

The local ``~/.amlhpc/<name>.sh`` profile that ``deploy init`` writes is the
*basis*: it records which datastore and prefix hold the site config
(``AMLHPC_CONFIG_DATASTORE`` / ``AMLHPC_CONFIG_PREFIX``), so the client knows
where to look. The blob is the source of truth; the profile only points at it.

Access uses the datastore's own credentials (account key or SAS) as returned by
``ml_client.datastores.get(..., include_secrets=True)``, so it works with the
plain Contributor role that ``deploy`` already grants -- no extra data-plane
blob RBAC is required.
"""

import os

DEFAULT_DATASTORE = "workspaceblobstore"
DEFAULT_PREFIX = "amlhpc"
PROLOG_BLOB = "prolog.sh"
EPILOG_BLOB = "epilog.sh"


def config_location():
    """Return (datastore, prefix) for the site config, honouring the profile.

    ``~/.amlhpc/<name>.sh`` (sourced into the environment) may override the
    defaults via AMLHPC_CONFIG_DATASTORE / AMLHPC_CONFIG_PREFIX.
    """
    datastore = os.environ.get("AMLHPC_CONFIG_DATASTORE", DEFAULT_DATASTORE)
    prefix = os.environ.get("AMLHPC_CONFIG_PREFIX", DEFAULT_PREFIX).strip("/")
    return datastore, prefix


def _container_client(ml_client, datastore):
    """Build an azure-storage-blob container client for a blob datastore.

    Uses the datastore's stored credentials (SAS or account key) so no separate
    data-plane RBAC assignment is needed.
    """
    from azure.storage.blob import BlobServiceClient

    ds = ml_client.datastores.get(datastore, include_secrets=True)
    account_name = getattr(ds, "account_name", None)
    container_name = getattr(ds, "container_name", None)
    if not account_name or not container_name:
        raise ValueError(
            "datastore '" + datastore + "' is not a blob datastore with an "
            "account/container; site config needs a blob datastore")
    endpoint = getattr(ds, "endpoint", None) or "core.windows.net"
    protocol = getattr(ds, "protocol", None) or "https"
    account_url = protocol + "://" + account_name + ".blob." + endpoint

    credential = None
    cred = getattr(ds, "credentials", None)
    account_key = getattr(cred, "account_key", None)
    sas_token = getattr(cred, "sas_token", None)
    if account_key:
        credential = account_key
    elif sas_token:
        credential = sas_token
    else:
        # Fall back to AAD; requires Storage Blob Data role on the account.
        from azure.identity import DefaultAzureCredential
        credential = DefaultAzureCredential()

    service = BlobServiceClient(account_url=account_url, credential=credential)
    return service.get_container_client(container_name)


def _read_blob(container, blob_path):
    """Return blob text, or None if it does not exist."""
    from azure.core.exceptions import ResourceNotFoundError

    try:
        data = container.get_blob_client(blob_path).download_blob().readall()
    except ResourceNotFoundError:
        return None
    return data.decode("utf-8")


def load_site_hooks(ml_client, verbose=0):
    """Return (prolog_text, epilog_text) from the storage stack.

    Missing blobs (or a missing/unreadable datastore) yield None for that hook;
    site config is optional, so read failures degrade to "no hook" rather than
    blocking job submission.
    """
    datastore, prefix = config_location()
    try:
        container = _container_client(ml_client, datastore)
    except Exception as error:
        if verbose:
            print("site config: could not open datastore '" + datastore +
                  "': " + str(error))
        return None, None

    base = (prefix + "/") if prefix else ""
    prolog = _read_blob(container, base + PROLOG_BLOB)
    epilog = _read_blob(container, base + EPILOG_BLOB)
    if verbose:
        print("site config datastore: " + datastore + ", prefix: " +
              (prefix or "<root>"))
        print("site prolog: " + ("found" if prolog else "none") +
              ", site epilog: " + ("found" if epilog else "none"))
    return prolog, epilog


def wrap_command_with_hooks(job_command, prolog, epilog):
    """Wrap a job command with site prolog/epilog, preserving its exit code.

    The prolog runs in the outer shell (so environment it sets up -- mounts,
    module loads -- is visible to both the command and the epilog). The user
    command runs in a subshell so that even a user ``exit`` cannot skip the
    epilog; its real exit status is captured and re-raised *after* the epilog,
    so a successful epilog cannot mask a failed job (and vice versa).
    """
    if not prolog and not epilog:
        return job_command

    parts = []
    if prolog:
        parts.append("# --- amlhpc site prolog ---")
        parts.append(prolog.rstrip("\n"))
    parts.append("__amlhpc_rc=0")
    parts.append("# --- user command ---")
    parts.append("(")
    parts.append(job_command)
    parts.append(") || __amlhpc_rc=$?")
    if epilog:
        parts.append("# --- amlhpc site epilog ---")
        parts.append(epilog.rstrip("\n"))
    parts.append("exit $__amlhpc_rc")
    return "\n".join(parts)


def apply_site_hooks(ml_client, job_command, enabled=True, verbose=0):
    """Convenience: load site hooks and wrap job_command, honouring an opt-out."""
    if not enabled:
        if verbose:
            print("site config: prolog/epilog disabled for this job")
        return job_command
    prolog, epilog = load_site_hooks(ml_client, verbose=verbose)
    return wrap_command_with_hooks(job_command, prolog, epilog)


def set_site_hook(ml_client, kind, text):
    """Upload a prolog/epilog snippet to the storage stack. kind: prolog|epilog."""
    datastore, prefix = config_location()
    blob = PROLOG_BLOB if kind == "prolog" else EPILOG_BLOB
    base = (prefix + "/") if prefix else ""
    container = _container_client(ml_client, datastore)
    container.get_blob_client(base + blob).upload_blob(
        text.encode("utf-8"), overwrite=True)
    return "azureml://datastores/" + datastore + "/paths/" + base + blob


def clear_site_hook(ml_client, kind):
    """Delete a prolog/epilog snippet from the storage stack."""
    from azure.core.exceptions import ResourceNotFoundError

    datastore, prefix = config_location()
    blob = PROLOG_BLOB if kind == "prolog" else EPILOG_BLOB
    base = (prefix + "/") if prefix else ""
    container = _container_client(ml_client, datastore)
    try:
        container.get_blob_client(base + blob).delete_blob()
        return True
    except ResourceNotFoundError:
        return False
