"""Tests for the PBS-style ``qsub`` front-end (amlhpc/pbs/qsub.py).

``qsub`` translates PBS options into an ``sbatch`` argv list and calls
``amlhpc.slurm.sbatch.sbatch`` (imported lazily at the end of the function).
We monkeypatch that target to capture the translated argv without touching
Azure.
"""

import pytest

import amlhpc.slurm.sbatch as sbatch_mod
from amlhpc.pbs.qsub import qsub


@pytest.fixture
def captured_sbatch(monkeypatch):
    """Capture the argv qsub/bsub hand to sbatch; sbatch itself is stubbed."""
    seen = {}

    def fake_sbatch(argv=None):
        seen["argv"] = argv
        return "fake_job_id"

    monkeypatch.setattr(sbatch_mod, "sbatch", fake_sbatch)
    return seen


def test_missing_queue_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        qsub(["hostname"])
    assert exc.value.code == -1
    assert "provide the queue to run the job with -q" in capsys.readouterr().out


def test_missing_command_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        qsub(["-q", "f16s"])
    assert exc.value.code == -1
    assert "provide a script to submit" in capsys.readouterr().out


def test_command_maps_to_wrap(captured_sbatch):
    rc = qsub(["-q", "f16s", "hostname"])
    assert rc == "fake_job_id"
    assert captured_sbatch["argv"] == ["-p", "f16s", "--wrap", "hostname"]


def test_multi_token_command_joined_into_wrap(captured_sbatch):
    qsub(["-q", "f16s", "echo", "hello", "world"])
    assert captured_sbatch["argv"] == ["-p", "f16s", "--wrap", "echo hello world"]


def test_nodes_resource_maps_to_nodes(captured_sbatch):
    qsub(["-q", "hbv2", "-l", "nodes=4", "hostname"])
    assert captured_sbatch["argv"] == ["-p", "hbv2", "-N", "4", "--wrap", "hostname"]


def test_select_resource_maps_to_nodes(captured_sbatch):
    qsub(["-q", "hbv2", "-l", "select=8", "hostname"])
    assert captured_sbatch["argv"] == ["-p", "hbv2", "-N", "8", "--wrap", "hostname"]


def test_nodes_ppn_resource_uses_node_count(captured_sbatch):
    qsub(["-q", "hbv2", "-l", "nodes=2:ppn=8", "hostname"])
    assert captured_sbatch["argv"] == ["-p", "hbv2", "-N", "2", "--wrap", "hostname"]


def test_existing_file_treated_as_script(captured_sbatch, tmp_path, monkeypatch):
    script = tmp_path / "runscript.sh"
    script.write_text("#!/bin/bash\nhostname\n")
    monkeypatch.chdir(tmp_path)
    qsub(["-q", "f16s", "runscript.sh"])
    # a real file becomes a positional script arg, not --wrap
    assert captured_sbatch["argv"] == ["-p", "f16s", "runscript.sh"]


def test_container_and_environment_pass_through(captured_sbatch):
    qsub(["-q", "f16s", "--container", "ubuntu:22.04", "-e", "myenv@latest", "hostname"])
    assert captured_sbatch["argv"] == [
        "-p", "f16s",
        "--container", "ubuntu:22.04",
        "-e", "myenv@latest",
        "--wrap", "hostname",
    ]


def test_name_is_informational_and_not_forwarded(captured_sbatch):
    qsub(["-q", "f16s", "-N", "myjob", "hostname"])
    # -N job name is informational for PBS; it must not leak into sbatch argv
    assert "myjob" not in captured_sbatch["argv"]
    assert captured_sbatch["argv"] == ["-p", "f16s", "--wrap", "hostname"]


def test_verbose_flag_forwarded(captured_sbatch):
    qsub(["-q", "f16s", "-v", "hostname"])
    assert captured_sbatch["argv"] == ["-p", "f16s", "-v", "--wrap", "hostname"]
