"""Tests for Bicep template resolution (amlhpc/deploy.py:resolve_template).

``resolve_template`` either validates a user-supplied ``--template`` path or
falls back to the template bundled in ``amlhpc.templates``, copying it to a
temp file the caller is responsible for deleting. Both paths are pure and
filesystem-only (no Azure), so they are exercised directly here.
"""

import os

import pytest

from amlhpc.deploy import resolve_template, write_cluster_profile


def test_explicit_existing_template_returned_without_tempfile(tmp_path):
    template = tmp_path / "custom.bicep"
    template.write_text("// bicep\n")
    resolved, tmp = resolve_template(str(template))
    # explicit path is returned as-is and no temp file is created to clean up
    assert resolved == str(template)
    assert tmp is None


def test_explicit_missing_template_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        resolve_template("/nonexistent/path/to/template.bicep")
    assert exc.value.code == -1
    out = capsys.readouterr().out
    assert "Bicep template '/nonexistent/path/to/template.bicep' not found" in out


def test_bundled_template_materialised_to_tempfile():
    resolved, tmp = resolve_template(None)
    try:
        # falls back to the packaged template, copied to a real temp file
        assert tmp is not None
        assert resolved == tmp
        assert os.path.isfile(resolved)
        assert resolved.endswith(".bicep")
        # the materialised file carries the bundled template's content
        content = open(resolved, "rb").read()
        assert content  # non-empty
        assert b"resource" in content or b"param" in content
    finally:
        if tmp and os.path.isfile(tmp):
            os.remove(tmp)


def test_write_cluster_profile_creates_sourceable_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    path = write_cluster_profile("amlhpc", "sub-123", "rg-hpc", "ml-workspace-xyz")

    assert path == str(tmp_path / ".amlhpc" / "amlhpc.sh")
    assert os.path.isfile(path)
    assert oct(os.stat(path).st_mode & 0o777) == "0o600"

    content = open(path).read()
    assert "export SUBSCRIPTION=sub-123" in content
    assert "export CI_RESOURCE_GROUP=rg-hpc" in content
    assert "export CI_WORKSPACE=ml-workspace-xyz" in content

