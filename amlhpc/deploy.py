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

    args = parser.parse_args(vargs)

    if args.subcommand == 'init':
        deploy_init(args)
    elif args.subcommand == 'partition':
        deploy_partition(args)
    elif args.subcommand == 'config':
        deploy_config(args)


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


def deploy_init(args):
    import os
    import shutil
    import subprocess

    if shutil.which('az') is None:
        print("deploy init requires the Azure CLI (az); install it or deploy the Bicep template manually (see deploy/README.md)")
        exit(-1)

    template, tmp_template = resolve_template(args.template)

    subprocess.run(["az", "group", "create", "--name", args.resource_group, "--location", args.location], check=True)

    deploy_command = ["az", "deployment", "group", "create",
                      "--resource-group", args.resource_group,
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


def deploy_partition(args):
    import os

    try:
        subscription_id = os.environ['SUBSCRIPTION']
        resource_group = os.environ['CI_RESOURCE_GROUP']
        workspace_name = os.environ['CI_WORKSPACE']
    except Exception as error:
        print("please set the export variables: SUBSCRIPTION, CI_RESOURCE_GROUP and CI_WORKSPACE")
        exit(-1)

    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
    from azure.ai.ml.entities import AmlCompute

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

    partition = AmlCompute(
        name=args.name,
        size=args.size,
        min_instances=args.min_nodes,
        max_instances=args.max_nodes,
        idle_time_before_scale_down=args.idle_time,
        tier=args.priority,
        )
    returned_partition = ml_client.compute.begin_create_or_update(partition).result()
    print(returned_partition.name + "\t" + returned_partition.size + "\t" + str(returned_partition.max_instances))


def config_ml_client():
    import os

    try:
        subscription_id = os.environ['SUBSCRIPTION']
        resource_group = os.environ['CI_RESOURCE_GROUP']
        workspace_name = os.environ['CI_WORKSPACE']
    except Exception:
        print("please set the export variables: SUBSCRIPTION, CI_RESOURCE_GROUP and CI_WORKSPACE")
        exit(-1)

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

