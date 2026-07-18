"""Tests for the unified ``amlhpc`` dispatcher (amlhpc/__main__.py)."""

from amlhpc.__main__ import main, _commands


def test_no_args_prints_usage_and_returns_error(capsys):
    rc = main([])
    out = capsys.readouterr().out
    assert rc == -1
    assert "usage: amlhpc <command> [<args>]" in out
    # every registered command is advertised in the help listing
    for name in _commands():
        assert name in out


def test_help_flag_returns_zero(capsys):
    for flag in ("-h", "--help"):
        rc = main([flag])
        out = capsys.readouterr().out
        assert rc == 0
        assert "commands:" in out


def test_unknown_command_reports_and_returns_error(capsys):
    rc = main(["bogus"])
    out = capsys.readouterr().out
    assert rc == -1
    assert "amlhpc: unknown command 'bogus'" in out
    assert "run 'amlhpc --help' for the list of commands" in out


def test_dispatch_routes_to_command_with_remaining_argv(monkeypatch):
    calls = {}

    def fake_qsub(argv):
        calls["argv"] = argv
        return 7

    # patch the entry actually stored in the dispatch table
    monkeypatch.setitem(main.__globals__, "_commands", lambda: {"qsub": fake_qsub})
    rc = main(["qsub", "-q", "f16s", "hostname"])
    assert rc == 7
    assert calls["argv"] == ["-q", "f16s", "hostname"]


def test_commands_table_is_complete():
    names = set(_commands())
    expected = {
        "sbatch", "srun", "sinfo", "squeue", "scancel",
        "sacct", "sstat", "sattach",
        "qsub", "qstat", "qdel", "bjobs", "bkill", "bsub",
        "container", "deploy",
        "dask-scheduler-up", "dask-up", "dask-down",
    }
    assert names == expected
    # every registered target is callable
    assert all(callable(fn) for fn in _commands().values())
