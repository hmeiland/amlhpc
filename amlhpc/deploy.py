class mlComputeAuth:
    def get_token(scopes="my_scope", claims="my_claim", tenant_id="my_tenant"):
        import requests
        import os
        from azure.core.credentials import AccessToken
        resource = "https://management.azure.com"
        client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID", None)
        resp = requests.get(f"{os.environ['MSI_ENDPOINT']}?resource={resource}&clientid={client_id}&api-version=2017-09-01", headers={'Secret': os.environ["MSI_SECRET"]})
        resp.raise_for_status()
        my_token = AccessToken(resp.json()["access_token"], int(resp.json()["expires_on"]))

        return my_token


def deploy(vargs=None):
    import argparse

    parser = argparse.ArgumentParser(description='deploy: provision amlhpc infrastructure on Azure Machine Learning')
    parser.prog = "deploy"
    subparsers = parser.add_subparsers(dest='subcommand', required=True)

    init_parser = subparsers.add_parser('init', help='deploy the initial amlhpc infrastructure from the Bicep template (workspace, dependencies, login VM and default cluster)')
    init_parser.add_argument('-g', '--resource-group', required=True, help='resource group to deploy into; created if it does not exist')
    init_parser.add_argument('-l', '--location', default='westus', help='Azure region for the resource group (default: westus)')
    init_parser.add_argument('-n', '--name', default='amlhpc', help='deployment name parameter, used as a prefix for the resource names (default: amlhpc)')
    init_parser.add_argument('-t', '--template', default=None, help='path to the Bicep template (default: the template bundled with amlhpc)')
    init_parser.add_argument('--enable-login-ssh', action='store_true', help='enable public SSH access on the login ComputeInstance (requires --login-ssh-key)')
    init_parser.add_argument('--login-ssh-key', default='', help='SSH public key (string or path to a .pub file) for the login CI admin user; required with --enable-login-ssh')
    init_parser.add_argument('--what-if', action='store_true', help='preview the changes without deploying any resources')

    partition_parser = subparsers.add_parser('partition', help='add a compute partition (AmlCompute cluster) to the workspace')
    partition_parser.add_argument('-n', '--name', required=True, help='partition (compute cluster) name')
    partition_parser.add_argument('-s', '--size', default='Standard_F2s_v2', help='VM size (default: Standard_F2s_v2)')
    partition_parser.add_argument('--min-nodes', default=0, type=int, help='minimum number of nodes to keep allocated (default: 0)')
    partition_parser.add_argument('--max-nodes', default=4, type=int, help='maximum number of nodes to scale up to (default: 4)')
    partition_parser.add_argument('--idle-time', default=120, type=int, help='seconds a node stays idle before scaling down (default: 120)')
    partition_parser.add_argument('--priority', default='Dedicated', choices=['Dedicated', 'LowPriority'], help='VM priority tier (default: Dedicated)')

    config_parser = subparsers.add_parser('config', help='manage the site-wide prolog/epilog stored in the workspace storage stack')
    config_parser.add_argument('action', choices=['set-prolog', 'set-epilog', 'show', 'clear-prolog', 'clear-epilog'], help='set-* uploads a snippet from FILE; show prints current hooks; clear-* removes one')
    config_parser.add_argument('file', nargs='?', default=None, help='shell snippet file to upload (required for set-prolog/set-epilog)')

    doctor_parser = subparsers.add_parser('doctor', help='check the workspace has everything amlhpc needs (default environment, a partition, login CI, datastore)')
    doctor_parser.add_argument('--fix', action='store_true', help='create the pieces that can be provisioned non-interactively (currently the default job environment)')

    connect_parser = subparsers.add_parser('connect', help='register an existing Azure Machine Learning workspace as a named cluster profile (~/.amlhpc/config.json)')
    connect_parser.add_argument('-n', '--name', required=True, help='local name for this cluster profile (used by --cluster / amlhpc use)')
    connect_parser.add_argument('-g', '--resource-group', required=True, help='resource group holding the workspace')
    connect_parser.add_argument('-w', '--workspace', required=True, help='Azure Machine Learning workspace name')
    connect_parser.add_argument('-s', '--subscription', default=None, help='subscription id (default: the SUBSCRIPTION environment variable)')
    connect_parser.add_argument('--no-current', action='store_true', help='register the profile without making it the current one')
    connect_parser.add_argument('--check', action='store_true', help="run 'deploy doctor' against the workspace after connecting")

    share_parser = subparsers.add_parser('share', help='export a cluster profile as a secret-free pointer another user can import (carries no credentials)')
    share_parser.add_argument('-n', '--name', required=True, help='name of the local cluster profile to export')
    share_parser.add_argument('-o', '--output', default=None, help='file to write the pointer to (default: stdout)')

    import_parser = subparsers.add_parser('import', help='import a cluster profile pointer produced by "deploy share"')
    import_parser.add_argument('-f', '--file', default=None, help='pointer file to read (default: stdin)')
    import_parser.add_argument('--no-current', action='store_true', help='import the profile without making it the current one')
    import_parser.add_argument('--check', action='store_true', help="run 'deploy doctor' against the workspace after importing")

    invite_parser = subparsers.add_parser('invite', help='grant a user access to the current workspace (AzureML Data Scientist role, workspace-scoped)')
    invite_parser.add_argument('user', help='user to grant access to: email/UPN or Azure AD object id')
    invite_parser.add_argument('--cluster', default=None, help='cluster profile to grant against (default: the resolved current cluster)')
    invite_parser.add_argument('--role', default='AzureML Data Scientist', help='role to assign (default: "AzureML Data Scientist")')
    invite_parser.add_argument('-y', '--yes', action='store_true', help='skip the confirmation prompt')

    uninvite_parser = subparsers.add_parser('uninvite', help='revoke a user\'s access to the current workspace (removes the workspace-scoped role assignment)')
    uninvite_parser.add_argument('user', help='user to revoke: email/UPN or Azure AD object id')
    uninvite_parser.add_argument('--cluster', default=None, help='cluster profile to revoke against (default: the resolved current cluster)')
    uninvite_parser.add_argument('--role', default='AzureML Data Scientist', help='role to remove (default: "AzureML Data Scientist")')
    uninvite_parser.add_argument('-y', '--yes', action='store_true', help='skip the confirmation prompt')

    validate_parser = subparsers.add_parser('validate', help='exercise every amlhpc feature end-to-end against the current cluster (submits a real EESSI job and watches it to completion)')
    validate_parser.add_argument('--cluster', default=None, help='cluster profile to validate (default: the resolved current cluster)')
    validate_parser.add_argument('-p', '--partition', default=None, help='partition (AmlCompute cluster) to submit the probe job to (default: the first AmlCompute partition doctor finds)')
    validate_parser.add_argument('--include-container', action='store_true', help='also build a container environment (needs a build context under --container-path)')
    validate_parser.add_argument('--container-path', default='environments', help='directory of container build contexts for --include-container (default: environments)')
    validate_parser.add_argument('--include-dask', action='store_true', help='also bring a Dask scheduler and worker up and back down on the partition')
    validate_parser.add_argument('--invite-user', default=None, help='email/UPN or object id to invite then uninvite, round-tripping the RBAC path')
    validate_parser.add_argument('--timeout', default=1800, type=int, help='seconds to wait for the probe job to reach a terminal state (default: 1800)')
    validate_parser.add_argument('--keep', action='store_true', help='do not cancel/clean the probe jobs created during validation')
    validate_parser.add_argument('-v', '--verbose', action='count', default=0, help='print each stage as it runs, not just the summary')

    args = parser.parse_args(vargs)

    if args.subcommand == 'init':
        deploy_init(args)
    elif args.subcommand == 'partition':
        deploy_partition(args)
    elif args.subcommand == 'config':
        deploy_config(args)
    elif args.subcommand == 'doctor':
        deploy_doctor(args)
    elif args.subcommand == 'connect':
        deploy_connect(args)
    elif args.subcommand == 'share':
        deploy_share(args)
    elif args.subcommand == 'import':
        deploy_import(args)
    elif args.subcommand == 'invite':
        deploy_invite(args)
    elif args.subcommand == 'uninvite':
        deploy_uninvite(args)
    elif args.subcommand == 'validate':
        deploy_validate(args)


def resolve_template(template):
    import os

    if template is not None:
        if not os.path.isfile(template):
            print("Missing: Bicep template '" + template + "' not found; check the path passed to --template")
            exit(-1)
        return template, None

    import importlib.resources
    import tempfile

    resource = importlib.resources.files('amlhpc.templates') / 'amlhpc_simple.bicep'
    tmp = tempfile.NamedTemporaryFile(prefix='amlhpc_simple_', suffix='.bicep', delete=False)
    tmp.write(resource.read_bytes())
    tmp.close()
    return tmp.name, tmp.name


def write_cluster_profile(name, subscription_id, resource_group, workspace_name):
    """Write ~/.amlhpc/<name>.sh with the exports needed to return to this cluster.

    Sourcing the file (`source ~/.amlhpc/<name>.sh`) restores SUBSCRIPTION,
    CI_RESOURCE_GROUP and CI_WORKSPACE so sbatch/dask-*/deploy partition work
    against the cluster again after coming back to it.
    """
    import os

    profile_dir = os.path.join(os.path.expanduser("~"), ".amlhpc")
    os.makedirs(profile_dir, exist_ok=True)
    profile_path = os.path.join(profile_dir, name + ".sh")

    with open(profile_path, "w") as profile:
        profile.write("# amlhpc cluster profile for '" + name + "'\n")
        profile.write("# source this file to return to the cluster: source " + profile_path + "\n")
        profile.write("export SUBSCRIPTION=" + subscription_id + "\n")
        profile.write("export CI_RESOURCE_GROUP=" + resource_group + "\n")
        profile.write("export CI_WORKSPACE=" + workspace_name + "\n")
        profile.write("# site-wide prolog/epilog live in the storage stack at <datastore>/<prefix>/{prolog,epilog}.sh\n")
        profile.write("export AMLHPC_CONFIG_DATASTORE=workspaceblobstore\n")
        profile.write("export AMLHPC_CONFIG_PREFIX=amlhpc\n")

    os.chmod(profile_path, 0o600)
    return profile_path


def deploy_init(args):
    import os
    import json
    import shutil
    import subprocess

    if shutil.which('az') is None:
        print("deploy init requires the Azure CLI (az); install it or deploy the Bicep template manually (see deploy/README.md)")
        exit(-1)

    template, tmp_template = resolve_template(args.template)

    # Resolve az to its full path: on Windows it is `az.cmd`, which a bareword
    # "az" cannot launch via subprocess (no shell).
    az = shutil.which('az')

    subprocess.run([az, "group", "create", "--name", args.resource_group, "--location", args.location], check=True)

    deploy_command = [az, "deployment", "group", "create",
                      "--resource-group", args.resource_group,
                      "--name", "amlhpc-" + args.name,
                      "--template-file", template,
                      "--parameters", "name=" + args.name]

    if args.enable_login_ssh:
        if not args.login_ssh_key:
            print("Missing: --enable-login-ssh requires --login-ssh-key (a public key string or path to a .pub file)")
            exit(-1)
        ssh_key = args.login_ssh_key
        if os.path.isfile(ssh_key):
            with open(ssh_key) as key_file:
                ssh_key = key_file.read().strip()
        deploy_command += ["--parameters", "enableLoginSsh=true", "--parameters", "loginSshPublicKey=" + ssh_key]

    if args.what_if:
        deploy_command.append("--what-if")
        try:
            subprocess.run(deploy_command, check=True)
        finally:
            if tmp_template is not None:
                os.remove(tmp_template)
        return

    deploy_command += ["--output", "json"]
    try:
        result = subprocess.run(deploy_command, check=True, capture_output=True, text=True)
    finally:
        if tmp_template is not None:
            os.remove(tmp_template)

    print(result.stdout)

    deployment = json.loads(result.stdout)
    workspace_name = deployment["properties"]["outputs"]["workspaceName"]["value"]
    subscription_id = deployment["id"].split("/")[2]

    profile_path = write_cluster_profile(args.name, subscription_id, args.resource_group, workspace_name)
    print("wrote cluster profile: " + profile_path)
    print("return to this cluster later with: source " + profile_path)

    from .context import put_profile
    config_path = put_profile(args.name, subscription_id, args.resource_group, workspace_name)
    print("registered cluster profile '" + args.name + "' in " + config_path
          + " (now the current cluster; amlhpc clusters to list, amlhpc use to switch)")


def deploy_partition(args):
    from .context import ConnectionNotConfigured, resolve_connection
    try:
        conn = resolve_connection()
    except ConnectionNotConfigured as error:
        print(error.message)
        exit(-1)
    subscription_id = conn.subscription
    resource_group = conn.resource_group
    workspace_name = conn.workspace

    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
    from azure.ai.ml.entities import AmlCompute, IdentityConfiguration

    import os
    import logging
    logging.getLogger('azure.ai.ml._utils').setLevel(logging.CRITICAL)

    try:
        on_aml = os.environ['APPSETTING_WEBSITE_SITE_NAME']
        if (on_aml == 'AMLComputeInstance'):
            credential = mlComputeAuth()
    except:
        credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
        enable_telemetry=False,
        )

    # A partition with identity: null cannot pull images from the workspace's
    # attached ACR (the executor stalls and the job fails), so give every
    # partition a SystemAssigned identity, matching the Bicep-provisioned ones.
    partition = AmlCompute(
        name=args.name,
        size=args.size,
        min_instances=args.min_nodes,
        max_instances=args.max_nodes,
        idle_time_before_scale_down=args.idle_time,
        tier=args.priority,
        identity=IdentityConfiguration(type="SystemAssigned"),
        )
    returned_partition = ml_client.compute.begin_create_or_update(partition).result()
    print(returned_partition.name + "\t" + returned_partition.size + "\t" + str(returned_partition.max_instances))


DEFAULT_ENVIRONMENT = "amlhpc-ubuntu2204"
DEFAULT_ENVIRONMENT_IMAGE = "docker.io/hmeiland/amlhpc-ubuntu2204"


def run_doctor_checks(ml_client):
    """Probe a workspace for the pieces amlhpc needs at runtime.

    Returns a list of check dicts: name, ok (bool), detail (human string),
    fixable (bool -- whether 'doctor --fix' can create it non-interactively).
    Kept pure (no printing, no Azure client construction) so it is unit
    testable against a stub ml_client and reused by both the report and --fix.

    The checks mirror real runtime dependencies:
      * default job environment 'amlhpc-ubuntu2204' -- sbatch falls back to it
        when no -e/--environment is given (see slurm/sbatch.py);
      * at least one AmlCompute partition -- sbatch errors without a -p target;
      * the default blob datastore 'workspaceblobstore' -- backs the site-wide
        prolog/epilog (see config.py);
      * a login ComputeInstance -- srun auto-discovers it when no -p is given.
    """
    checks = []

    have_env = False
    try:
        for env in ml_client.environments.list(name=DEFAULT_ENVIRONMENT):
            have_env = True
            break
    except Exception:
        have_env = _any_named_environment(ml_client, DEFAULT_ENVIRONMENT)
    checks.append({
        "name": "default job environment '" + DEFAULT_ENVIRONMENT + "'",
        "ok": have_env,
        "detail": ("present" if have_env
                   else "missing; sbatch has no default environment to fall back on"),
        "fixable": True,
    })

    computes = list(_safe_list(ml_client.compute.list))
    partitions = [c for c in computes
                  if str(getattr(c, "type", "")).lower() == "amlcompute"]
    instances = [c for c in computes
                 if str(getattr(c, "type", "")).lower() == "computeinstance"]

    checks.append({
        "name": "at least one AmlCompute partition",
        "ok": bool(partitions),
        "detail": (", ".join(sorted(c.name for c in partitions))
                   if partitions
                   else "none found; add one with 'deploy partition -n <name> -s <size>'"),
        "fixable": False,
    })

    checks.append({
        "name": "login ComputeInstance (for srun)",
        "ok": bool(instances),
        "detail": (", ".join(sorted(c.name for c in instances))
                   if instances
                   else "none found; srun has no login node to auto-discover"),
        "fixable": False,
    })

    have_datastore = False
    try:
        ml_client.datastores.get("workspaceblobstore")
        have_datastore = True
    except Exception:
        have_datastore = False
    checks.append({
        "name": "default datastore 'workspaceblobstore'",
        "ok": have_datastore,
        "detail": ("present" if have_datastore
                   else "missing; site prolog/epilog storage is unavailable"),
        "fixable": False,
    })

    return checks


def _safe_list(list_callable):
    """Call an SDK list() and return an iterable, tolerating transport errors."""
    try:
        return list(list_callable())
    except Exception:
        return []


def _any_named_environment(ml_client, name):
    """Fallback existence probe for SDKs whose list() rejects a name kwarg."""
    try:
        for env in ml_client.environments.list():
            if getattr(env, "name", None) == name:
                return True
    except Exception:
        return False
    return False


def create_default_environment(ml_client):
    """Create the default 'amlhpc-ubuntu2204' job environment. Returns its name."""
    from azure.ai.ml.entities import Environment

    environment = Environment(
        name=DEFAULT_ENVIRONMENT,
        image=DEFAULT_ENVIRONMENT_IMAGE,
        description="Default amlhpc job environment (created by 'deploy doctor --fix').",
    )
    created = ml_client.environments.create_or_update(environment)
    return getattr(created, "name", DEFAULT_ENVIRONMENT)


def deploy_connect(args):
    """Register an existing workspace as a named profile in ~/.amlhpc/config.json.

    This is the non-provisioning counterpart to 'deploy init': the workspace
    already exists (someone else ran init, or it predates amlhpc), and the user
    just wants their client to talk to it. Subscription defaults to $SUBSCRIPTION
    because that is the one identifier a user reliably has exported already.
    """
    import os

    from .context import put_profile

    subscription = args.subscription or os.environ.get('SUBSCRIPTION')
    if not subscription:
        print("Missing: pass -s/--subscription or export SUBSCRIPTION")
        exit(-1)

    path = put_profile(args.name, subscription, args.resource_group,
                       args.workspace, make_current=not args.no_current)
    print("registered cluster profile '" + args.name + "' in " + path)
    if not args.no_current:
        print("it is now the current cluster (amlhpc clusters to list, amlhpc use to switch)")

    if args.check:
        deploy_doctor(_DoctorArgs(cluster=args.name))


class _DoctorArgs:
    """Minimal args object so deploy_connect can invoke the doctor in-process."""

    def __init__(self, cluster=None, fix=False):
        self.cluster = cluster
        self.fix = fix


def build_role_command(action, az_path, role, scope, user):
    """Build the 'az role assignment <action>' argv for a workspace-scoped grant.

    Kept pure (no side effects) so the exact command can be unit tested and
    reused verbatim by both the executed path and the printed-fallback path.
    ``action`` is 'create' or 'delete'.
    """
    return [az_path, "role", "assignment", action,
            "--role", role,
            "--scope", scope,
            "--assignee", user]


def _run_role_assignment(action, verb, args):
    """Shared body for invite/uninvite: resolve, confirm, run 'az', or fall back.

    Access is separate from the shared profile: the profile only names the
    workspace, so granting a role via Azure RBAC is what actually lets an
    invited user reach it. If 'az' is absent or the caller lacks permission to
    change role assignments, the exact command is printed for an admin to run.
    """
    import shutil
    import subprocess

    from .context import ConnectionNotConfigured, resolve_connection

    try:
        conn = resolve_connection(args.cluster)
    except ConnectionNotConfigured as error:
        print(error.message)
        exit(-1)

    scope = conn.workspace_uri()
    az = shutil.which('az')
    command = build_role_command(action, az or 'az', args.role, scope, args.user)
    printable = " ".join(('"' + part + '"' if ' ' in part else part)
                         for part in command)

    if az is None:
        print("the Azure CLI (az) is not installed; ask an admin to run:")
        print("  " + printable)
        return

    print(verb + " '" + args.user + "' " + ("to" if action == "create" else "from")
          + " workspace '" + conn.workspace + "' with role '" + args.role + "'")
    if not args.yes:
        answer = input("proceed? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("aborted; no changes made")
            return

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        print(("granted" if action == "create" else "revoked") + " access for '"
              + args.user + "'")
        return

    print(verb + " failed (az exit " + str(result.returncode) + "):")
    if result.stderr:
        print(result.stderr.strip())
    print("if you lack permission to manage role assignments, ask an admin to run:")
    print("  " + printable)
    exit(1)


def deploy_invite(args):
    """Grant a user workspace-scoped access to the current cluster."""
    _run_role_assignment("create", "inviting", args)


def deploy_uninvite(args):
    """Revoke a user's workspace-scoped access to the current cluster."""
    _run_role_assignment("delete", "revoking", args)


def deploy_share(args):
    """Print or write a secret-free pointer to a cluster profile.

    The pointer carries only the workspace identifiers -- never credentials --
    so it is safe to send over chat or email. The recipient still authenticates
    as themselves and needs their own Azure access ('deploy invite' grants it).
    """
    import json

    from .context import ConnectionNotConfigured, export_profile

    try:
        blob = export_profile(args.name)
    except ConnectionNotConfigured as error:
        print(error.message)
        exit(-1)

    text = json.dumps(blob, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text + "\n")
        print("wrote shareable pointer for '" + args.name + "' to " + args.output)
    else:
        print(text)


def deploy_import(args):
    """Import a pointer from 'deploy share' (file or stdin) into the local store."""
    import json
    import sys

    from .context import ConnectionNotConfigured, import_profile

    if args.file:
        with open(args.file) as handle:
            raw = handle.read()
    else:
        raw = sys.stdin.read()

    try:
        blob = json.loads(raw)
    except ValueError:
        print("could not parse the pointer as JSON; expected the output of 'deploy share'")
        exit(-1)

    try:
        name = import_profile(blob, make_current=not args.no_current)
    except ConnectionNotConfigured as error:
        print(error.message)
        exit(-1)

    print("imported cluster profile '" + name + "'")
    if not args.no_current:
        print("it is now the current cluster (amlhpc use to switch, amlhpc clusters to list)")

    if args.check:
        deploy_doctor(_DoctorArgs(cluster=name))


def deploy_doctor(args):
    """Report (and optionally --fix) the workspace's amlhpc feature-completeness."""
    ml_client = config_ml_client(getattr(args, "cluster", None))
    checks = run_doctor_checks(ml_client)

    print("amlhpc feature-completeness check")
    all_ok = True
    for check in checks:
        marker = "[ok]  " if check["ok"] else "[FAIL]"
        print(marker + " " + check["name"] + ": " + check["detail"])
        if not check["ok"]:
            all_ok = False

    if all_ok:
        print("\nworkspace is amlhpc-ready")
        return

    fixable = [c for c in checks if not c["ok"] and c["fixable"]]
    if not getattr(args, "fix", False):
        if fixable:
            print("\nre-run with --fix to create: "
                  + ", ".join(c["name"] for c in fixable))
        exit(1)

    if not fixable:
        print("\nnothing here can be fixed automatically; see the guidance above")
        exit(1)

    for check in fixable:
        if check["name"].startswith("default job environment"):
            name = create_default_environment(ml_client)
            print("created default environment: " + name)

    remaining = [c for c in run_doctor_checks(ml_client) if not c["ok"]]
    if remaining:
        print("\nstill incomplete (needs manual action): "
              + ", ".join(c["name"] for c in remaining))
        exit(1)
    print("\nworkspace is now amlhpc-ready")


def config_ml_client(cluster=None):
    import os

    from .context import ConnectionNotConfigured, resolve_connection
    try:
        conn = resolve_connection(cluster)
    except ConnectionNotConfigured as error:
        print(error.message)
        exit(-1)
    subscription_id = conn.subscription
    resource_group = conn.resource_group
    workspace_name = conn.workspace

    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    import logging
    logging.getLogger('azure.ai.ml._utils').setLevel(logging.CRITICAL)

    try:
        on_aml = os.environ['APPSETTING_WEBSITE_SITE_NAME']
        if (on_aml == 'AMLComputeInstance'):
            credential = mlComputeAuth()
    except Exception:
        credential = DefaultAzureCredential()

    return MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
        enable_telemetry=False,
        )


def deploy_config(args):
    from amlhpc import config

    ml_client = config_ml_client()
    datastore, prefix = config.config_location()

    if args.action in ('set-prolog', 'set-epilog'):
        if not args.file:
            print("Missing: " + args.action + " requires a FILE holding the shell snippet")
            exit(-1)
        with open(args.file) as snippet:
            text = snippet.read()
        kind = 'prolog' if args.action == 'set-prolog' else 'epilog'
        uri = config.set_site_hook(ml_client, kind, text)
        print("uploaded site " + kind + " to " + uri)
        return

    if args.action in ('clear-prolog', 'clear-epilog'):
        kind = 'prolog' if args.action == 'clear-prolog' else 'epilog'
        removed = config.clear_site_hook(ml_client, kind)
        print(("removed" if removed else "no") + " site " + kind +
              " in " + datastore + "/" + prefix)
        return

    prolog, epilog = config.load_site_hooks(ml_client)
    print("site config datastore: " + datastore + ", prefix: " + prefix)
    for kind, text in (("prolog", prolog), ("epilog", epilog)):
        print("\n===== " + kind + " =====")
        print(text if text else "(none)")



# EESSI cvmfs software stack used by the probe job (see examples/EESSI/readme.md).
# The amlhpc job container runs privileged with FUSE, so a job can mount EESSI,
# load a module and run a real application on the AmlCompute node.
EESSI_INIT = ("sudo mount -t cvmfs software.eessi.io /cvmfs/software.eessi.io; "
              "source /cvmfs/software.eessi.io/versions/2023.06/init/bash")

# AML job states that mean the job has stopped (no further transitions).
TERMINAL_JOB_STATES = frozenset({"completed", "failed", "canceled", "cancelled"})


def build_eessi_wrap(command):
    """Build the sbatch --wrap body that mounts EESSI then runs ``command``.

    Kept pure so the probe job's command line can be unit tested without
    submitting anything.
    """
    return EESSI_INIT + "; " + command


def is_terminal_state(status):
    """True when an AML job status means the job has stopped.

    Tolerant of case and of None (a job with no status yet is not terminal).
    Kept pure so the watch loop's stop condition is unit testable.
    """
    return str(status or "").strip().lower() in TERMINAL_JOB_STATES


class ValidationReport:
    """Records the pass/fail/skip outcome of each validation stage.

    Pure (no printing, no Azure) so the orchestration in deploy_validate can be
    unit tested by inspecting the recorded stages, mirroring run_doctor_checks.
    """

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"

    def __init__(self):
        self.stages = []

    def record(self, outcome, name, detail=""):
        self.stages.append({"outcome": outcome, "name": name, "detail": detail})

    def ok(self, name, detail=""):
        self.record(self.PASS, name, detail)

    def fail(self, name, detail=""):
        self.record(self.FAIL, name, detail)

    def skip(self, name, detail=""):
        self.record(self.SKIP, name, detail)

    @property
    def failed(self):
        return [s for s in self.stages if s["outcome"] == self.FAIL]

    @property
    def all_ok(self):
        return not self.failed


def _watch_job(ml_client, jobid, timeout, poll=15):
    """Poll a job until it reaches a terminal state or ``timeout`` seconds pass.

    Returns the final status string (possibly non-terminal if it timed out).
    """
    import time

    deadline = time.time() + timeout
    status = None
    while True:
        job = ml_client.jobs.get(jobid)
        status = getattr(job, "status", None)
        if is_terminal_state(status) or time.time() >= deadline:
            return status
        time.sleep(poll)


def _job_in_listing(ml_client, jobid, attempts=5, poll=6):
    """True if ``jobid`` appears in jobs.list(), retrying to absorb index lag.

    jobs.get() is a direct lookup that resolves a just-submitted job at once,
    but jobs.list() is an eventually-consistent index that can trail submission
    by seconds; a bounded retry mirrors a user re-running squeue a moment later.
    """
    import time

    for attempt in range(attempts):
        for page in ml_client.jobs.list().by_page():
            for job in page:
                if getattr(job, "name", None) == jobid:
                    return True
        if attempt < attempts - 1:
            time.sleep(poll)
    return False


def deploy_validate(args):
    """Exercise every amlhpc feature end-to-end against the current cluster.

    Runs a staged sequence of real operations -- doctor, profile listing,
    partition discovery, a real EESSI probe job watched to completion, the job
    introspection/cancel front-ends against that job's real JOBID, srun on the
    login CI, and the opt-in container/dask/RBAC paths -- recording each stage's
    outcome and exiting non-zero if any stage failed. Reuses the same building
    blocks the real commands use (run_doctor_checks, get_ml_client, the slurm
    front-ends) so a green run means the shipped tools work against this
    workspace, not a mock of it.
    """
    import time

    report = ValidationReport()
    verbose = getattr(args, "verbose", 0)

    def announce(text):
        if verbose:
            print(text)

    from .context import ConnectionNotConfigured, resolve_connection
    try:
        conn = resolve_connection(getattr(args, "cluster", None))
    except ConnectionNotConfigured as error:
        print(error.message)
        exit(-1)
    announce("[..] connection: " + conn.workspace)
    report.ok("connection", "workspace '" + conn.workspace + "'")

    ml_client = config_ml_client(getattr(args, "cluster", None))

    checks = run_doctor_checks(ml_client)
    failing = [c for c in checks if not c["ok"]]
    if failing:
        report.fail("doctor", "; ".join(c["name"] for c in failing))
        announce("[!!] doctor: " + ", ".join(c["name"] for c in failing))
    else:
        report.ok("doctor", "all checks pass")
        announce("[ok] doctor")

    partition = getattr(args, "partition", None)
    if partition is None:
        computes = list(_safe_list(ml_client.compute.list))
        partitions = [c for c in computes
                      if str(getattr(c, "type", "")).lower() == "amlcompute"]
        partition = sorted(c.name for c in partitions)[0] if partitions else None

    from .context import load_config, use_profile
    config = load_config()
    current = config.get("current")
    if current:
        try:
            use_profile(current)
            report.ok("clusters/use", "current '" + current + "'")
        except ConnectionNotConfigured as error:
            report.fail("clusters/use", error.message)
    else:
        report.skip("clusters/use", "no named current profile (using env/explicit)")

    try:
        computes = list(_safe_list(ml_client.compute.list))
        names = sorted(str(getattr(c, "name", "")) for c in computes)
        if partition and partition in names:
            report.ok("sinfo", "partition '" + partition + "' listed")
        elif partition:
            report.fail("sinfo", "partition '" + partition + "' not in " + ", ".join(names))
        else:
            report.skip("sinfo", "no AmlCompute partition found to submit to")
    except Exception as error:
        report.fail("sinfo", str(error))

    if partition is None:
        report.skip("submit", "no partition; skipping job-dependent stages")
        _print_validation_report(report)
        if not report.all_ok:
            exit(1)
        return

    jobid = None
    from .slurm.sbatch import sbatch
    wrap = build_eessi_wrap("ml load OpenFOAM; source $FOAM_BASH; simpleFoam -help")
    try:
        jobid = _capture_stdout_token(
            lambda: sbatch(['-p', partition, '--wrap', wrap]))
        if jobid:
            report.ok("sbatch (EESSI)", "JOBID " + jobid)
            announce("[ok] sbatch: " + jobid)
        else:
            report.fail("sbatch (EESSI)", "no JOBID printed")
    except SystemExit as error:
        report.fail("sbatch (EESSI)", "exit " + str(error.code))
    except Exception as error:
        report.fail("sbatch (EESSI)", str(error))

    if jobid:
        try:
            found = _job_in_listing(ml_client, jobid)
            (report.ok if found else report.fail)(
                "squeue/qstat/bjobs", ("job listed" if found else "job not in listing"))
        except Exception as error:
            report.fail("squeue/qstat/bjobs", str(error))

        try:
            ml_client.jobs.get(jobid)
            report.ok("sacct", "status for " + jobid)
        except Exception as error:
            report.fail("sacct", str(error))

        announce("[..] watching " + jobid + " (timeout " + str(args.timeout) + "s)")
        try:
            final = _watch_job(ml_client, jobid, args.timeout)
            if is_terminal_state(final):
                (report.ok if str(final).lower() == "completed" else report.fail)(
                    "job watch", "final state " + str(final))
            else:
                report.fail("job watch", "did not finish within " + str(args.timeout) + "s (last: " + str(final) + ")")
        except Exception as error:
            report.fail("job watch", str(error))

        try:
            from .jobcontrol import attach_job
            _capture_stdout_token(lambda: attach_job("sattach", [jobid]))
            report.ok("sattach", "log fetched")
        except SystemExit:
            report.fail("sattach", "no log available")
        except Exception as error:
            report.fail("sattach", str(error))

        try:
            from .jobcontrol import show_job_stats
            _capture_stdout_token(lambda: show_job_stats("sstat", [jobid]))
            report.ok("sstat", "utilization queried")
        except SystemExit as error:
            report.fail("sstat", "exit " + str(error.code))
        except Exception as error:
            report.fail("sstat", str(error))

    srun_jobid = None
    try:
        from .slurm.srun import srun
        srun_jobid = _capture_stdout_token(lambda: srun(['--wrap', 'hostname']))
        if srun_jobid:
            _watch_job(ml_client, srun_jobid, args.timeout)
            report.ok("srun", "login-CI job " + srun_jobid)
        else:
            report.fail("srun", "no JOBID printed")
    except SystemExit as error:
        report.fail("srun", "exit " + str(error.code))
    except Exception as error:
        report.fail("srun", str(error))

    if args.include_container:
        import os
        if not os.path.isdir(args.container_path):
            report.skip("container", "no build contexts under '" + args.container_path + "'")
        else:
            try:
                from .container import container
                _capture_stdout_token(
                    lambda: container(['-p', args.container_path]))
                report.ok("container", "environment(s) built")
            except SystemExit as error:
                report.fail("container", "exit " + str(error.code))
            except Exception as error:
                report.fail("container", str(error))
    else:
        report.skip("container", "not requested (--include-container)")

    if args.include_dask:
        report.skip("dask", "dask up/down requires an interactive session; run manually with dask-scheduler-up/dask-up/dask-down")
    else:
        report.skip("dask", "not requested (--include-dask)")

    if args.invite_user:
        try:
            _run_role_assignment("create", "inviting",
                                 _InviteArgs(args.invite_user, cluster=args.cluster))
            _run_role_assignment("delete", "revoking",
                                 _InviteArgs(args.invite_user, cluster=args.cluster))
            report.ok("invite/uninvite", "round-tripped '" + args.invite_user + "'")
        except SystemExit as error:
            report.fail("invite/uninvite", "exit " + str(error.code))
        except Exception as error:
            report.fail("invite/uninvite", str(error))
    else:
        report.skip("invite/uninvite", "not requested (--invite-user)")

    if not args.keep:
        try:
            from .slurm.sbatch import sbatch as _sbatch
            from .jobcontrol import cancel_job
            throwaway = _capture_stdout_token(
                lambda: _sbatch(['-p', partition, '--wrap', 'sleep 600']))
            if throwaway:
                time.sleep(2)
                _capture_stdout_token(lambda: cancel_job("scancel", [throwaway]))
                report.ok("scancel", "cancelled " + throwaway)
            else:
                report.fail("scancel", "could not submit throwaway job")
        except SystemExit as error:
            report.fail("scancel", "exit " + str(error.code))
        except Exception as error:
            report.fail("scancel", str(error))
    else:
        report.skip("scancel", "--keep: left probe jobs running")

    _print_validation_report(report)
    if not report.all_ok:
        exit(1)


class _InviteArgs:
    """Minimal args object so deploy_validate can invoke the RBAC path in-process."""

    def __init__(self, user, cluster=None, role="AzureML Data Scientist", yes=True):
        self.user = user
        self.cluster = cluster
        self.role = role
        self.yes = yes


def _capture_stdout_token(call):
    """Run ``call`` (a command front-end), echo its stdout, return the last token.

    The job front-ends print the JOBID as their only stdout line on success; we
    still surface everything they print (so the run is auditable) but hand back
    the final whitespace token as the JOBID for the stages that need it.
    """
    import io
    import sys

    buffer = io.StringIO()
    saved = sys.stdout
    sys.stdout = buffer
    try:
        call()
    finally:
        sys.stdout = saved

    text = buffer.getvalue()
    if text:
        print(text, end="" if text.endswith("\n") else "\n")
    tokens = text.split()
    return tokens[-1] if tokens else None


def _print_validation_report(report):
    print("\namlhpc end-to-end validation")
    marker = {report.PASS: "[ok]  ", report.FAIL: "[FAIL]", report.SKIP: "[skip]"}
    for stage in report.stages:
        line = marker[stage["outcome"]] + " " + stage["name"]
        if stage["detail"]:
            line += ": " + stage["detail"]
        print(line)

    passed = sum(1 for s in report.stages if s["outcome"] == report.PASS)
    failed = len(report.failed)
    skipped = sum(1 for s in report.stages if s["outcome"] == report.SKIP)
    print("\n" + str(passed) + " passed, " + str(failed) + " failed, "
          + str(skipped) + " skipped")
    if report.all_ok:
        print("workspace validated: amlhpc features work end-to-end")
    else:
        print("validation FAILED: " + ", ".join(s["name"] for s in report.failed))

