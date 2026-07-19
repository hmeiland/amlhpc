# GUI / VDI access

amlhpc is a stateless client: any authenticated shell that can reach Azure can submit jobs.
A "VDI" is therefore just that client with a graphical surface. This example walks through the
two supported ways to get one, plus a validated Windows-workstation path.

See [architecture.md](../../architecture.md) (*GUI / VDI access*) for the design rationale. The
identity model is unchanged throughout: the login ComputeInstance's **system-assigned managed
identity** with **Contributor on the resource group** (the `roleAssignment` created by
`deploy init`). GUI sessions authenticate through `DefaultAzureCredential` / the CI MSI exactly
like the CLI — no keys, no per-user service principals.

## Option A — built-in browser apps on the login ComputeInstance (recommended, zero extra deploy)

The login node that `deploy init` creates is an AML ComputeInstance, and every ComputeInstance
already publishes JupyterLab, Jupyter, VS Code (browser + desktop), RStudio/Posit and an in-browser
terminal, brokered by AML studio over authenticated HTTPS (no public IP, no inbound port, no SSH).

1. Open [ml.azure.com](https://ml.azure.com), select the workspace, go to **Compute → Compute
   instances**, and start the login CI (name `login-<suffix>`).
2. Click **Terminal** (or **JupyterLab → Terminal**).
3. The Bicep bakes `SUBSCRIPTION` into `/etc/profile.d/amlhpc.sh` and AML sets
   `CI_RESOURCE_GROUP` / `CI_WORKSPACE`, so the CLI works with no login:
   ```
   sinfo
   sbatch -p f4s --wrap="hostname"
   squeue
   ```

That is the zero-cost VDI: a job submitted from a GUI, authenticated by the CI's MSI, exactly like
`srun`. For notebook/editor-driven workflows nothing else is needed.

## Option B — a full graphical desktop (custom application)

For GUI applications that are not a notebook or editor (a native pre/post-processor such as
ParaView, a graphical file manager, a browser), host a **custom application** on the login CI: a
container image, referenced from the CI's `applications` block, that AML reverse-proxies over the
same authenticated HTTPS channel. A noVNC/XFCE (or KASM) desktop image gives a full Linux desktop
in a browser tab.

Add a `desktopApp` entry to the `amlLoginVM` ComputeInstance in
[`amlhpc/templates/amlhpc_simple.bicep`](../../amlhpc/templates/amlhpc_simple.bicep) (illustrative):

```bicep
properties: {
  // ...existing login CI properties...
  applications: [
    {
      displayName: 'desktop'
      endpointUri: 'http://localhost:6901/'          // port the desktop image serves noVNC on
    }
  ]
}
```

Requirements for the image:

- amlhpc `pip install`-ed into it (or mounted from the CI), so a terminal inside the desktop
  submits jobs through the same identity;
- it serves its web UI on the `endpointUri` port; AML fronts it with the authenticated proxy.

For a heavier workstation, create a **separate, larger ComputeInstance** alongside the small login
CI rather than growing the login node. It inherits the same identity/networking story.

## Option C — a Windows workstation client (validated)

Any Windows machine with the Azure CLI, Python and amlhpc can submit jobs; it authenticates with
`az login` (`DefaultAzureCredential`). This path was validated end-to-end from Windows PowerShell —
`deploy init` → `deploy partition` → `sbatch` → job `Completed` on an `Standard_F4s_v2` node.

```powershell
winget install --id Python.Python.3.12 -e
python -m venv amlhpc-venv
.\amlhpc-venv\Scripts\Activate.ps1
pip install amlhpc
az login
$env:SUBSCRIPTION      = "<subscription-guid>"
$env:CI_RESOURCE_GROUP = "<resource-group>"
$env:CI_WORKSPACE      = "<workspace-name>"
sinfo
sbatch -p f4s --wrap="hostname"
squeue
```

Because this is a self-managed resource outside the workspace, prefer it only when a genuinely
Windows-only GUI tool is required (see architecture.md). Two Windows-specific portability fixes
landed alongside this example:

- `deploy init` resolves `az` to its full path (on Windows the CLI is `az.cmd`, which a bareword
  `"az"` cannot launch via `subprocess`);
- `sbatch` falls back to `os.getcwd()` when the POSIX-only `PWD` environment variable is absent.
