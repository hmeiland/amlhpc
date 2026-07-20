"""amlhpc - Emulate Slurm/PBS/LSF HPC scheduler in Azure ML.

Author:
    Hugo Meiland <hugo.meiland@microsoft.com>

Contributors:
    Utkarsh Ayachit <uayachit@microsoft.com>
"""

import logging as _logging

# The azure.ai.ml SDK emits a WARNING for every "experimental" schema class it
# instantiates (e.g. "Class AutoDeleteSettingSchema: This is an experimental
# class ..."). These are pure SDK noise for our CLI users. Raising the emitting
# logger to ERROR here -- at package import, before any azure.ai.ml import --
# silences them across every entry point.
_logging.getLogger("azure.ai.ml._utils._experimental").setLevel(_logging.ERROR)
