"""Tests for the LSF-style ``bsub`` front-end (amlhpc/lsf/bsub.py).

``bsub`` translates LSF options into an ``sbatch`` argv list and calls
``amlhpc.slurm.sbatch.sbatch`` (imported lazily at the end). We stub that
target to capture the translated argv without touching Azure.
"""

import pytest

import amlhpc.slurm.sbatch as sbatch_mod
from amlhpc.lsf.bsub import bsub


@pytest.fixture
def captured_sbatch(monkeypatch):
    seen = {}

    def fake_sbatch(argv=None):
        seen["argv"] = argv
        return "fake_job_id"

    monkeypatch.setattr(sbatch_mod, "sbatch", fake_sbatch)
    return seen


def test_missing_queue_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        bsub(["hostname"])
    assert exc.value.code == -1
    assert "provide the queue to run the job with -q" in capsys.readouterr().out


def test_missing_command_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        bsub(["-q", "f16s"])
    assert exc.value.code == -1
    assert "provide the command (or script) to execute" in capsys.readouterr().out


def test_command_maps_to_wrap(captured_sbatch):
    rc = bsub(["-q", "f16s", "hostname"])
    assert rc == "fake_job_id"
    assert captured_sbatch["argv"] == ["-p", "f16s", "--wrap", "hostname"]


def test_multi_token_command_joined_into_wrap(captured_sbatch):
    bsub(["-q", "f16s", "echo", "hi", "there"])
    assert captured_sbatch["argv"] == ["-p", "f16s", "--wrap", "echo hi there"]


def test_num_slots_maps_to_nodes(captured_sbatch):
    bsub(["-q", "hbv2", "-n", "4", "hostname"])
    assert captured_sbatch["argv"] == ["-p", "hbv2", "-N", "4", "--wrap", "hostname"]


def test_existing_file_treated_as_script(captured_sbatch, tmp_path, monkeypatch):
    script = tmp_path / "runscript.sh"
    script.write_text("#!/bin/bash\nhostname\n")
    monkeypatch.chdir(tmp_path)
    bsub(["-q", "f16s", "runscript.sh"])
    assert captured_sbatch["argv"] == ["-p", "f16s", "runscript.sh"]


def test_container_and_environment_pass_through(captured_sbatch):
    bsub(["-q", "f16s", "--container", "ubuntu:22.04", "-e", "myenv@latest", "hostname"])
    assert captured_sbatch["argv"] == [
        "-p", "f16s",
        "--container", "ubuntu:22.04",
        "-e", "myenv@latest",
        "--wrap", "hostname",
    ]


def test_job_name_is_informational_and_not_forwarded(captured_sbatch):
    bsub(["-q", "f16s", "-J", "myjob", "hostname"])
    assert "myjob" not in captured_sbatch["argv"]
    assert captured_sbatch["argv"] == ["-p", "f16s", "--wrap", "hostname"]
