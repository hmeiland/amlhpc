"""Shared fixtures for the amlhpc test suite.

The command front-ends read AML connection settings from the environment and,
when they actually talk to Azure, build an ``MLClient`` at import/run time. The
tests here exercise only the pure, deterministic logic (argv translation,
dispatch, validation, template resolution), so we scrub the AML environment
variables to keep behaviour reproducible regardless of the developer's shell.
"""

import pytest

_AML_ENV_VARS = (
    "SUBSCRIPTION",
    "CI_RESOURCE_GROUP",
    "CI_WORKSPACE",
    "APPSETTING_WEBSITE_SITE_NAME",
    "MSI_ENDPOINT",
    "MSI_SECRET",
    "DEFAULT_IDENTITY_CLIENT_ID",
)


@pytest.fixture(autouse=True)
def _clean_aml_env(monkeypatch):
    for name in _AML_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
