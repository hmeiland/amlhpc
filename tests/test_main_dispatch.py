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
        "use", "clusters",
    }
    assert names == expected
    # every registered target is callable
    assert all(callable(fn) for fn in _commands().values())


def test_clusters_lists_profiles_and_marks_current(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    from amlhpc.context import put_profile
    put_profile("prod", "sub-1", "rg-1", "ws-prod")
    put_profile("dev", "sub-2", "rg-2", "ws-dev", make_current=False)
    rc = main(["clusters"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ws-prod" in out and "ws-dev" in out
    # the current profile (prod) is flagged, dev is not
    prod_line = [ln for ln in out.splitlines() if "prod" in ln][0]
    dev_line = [ln for ln in out.splitlines() if " dev " in ln or ln.strip().endswith("ws-dev")][0]
    assert "*" in prod_line
    assert "*" not in dev_line


def test_clusters_empty_gives_hint(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    rc = main(["clusters"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no cluster profiles yet" in out


def test_use_switches_current_profile(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    from amlhpc.context import put_profile, load_config
    put_profile("prod", "sub-1", "rg-1", "ws-prod")
    put_profile("dev", "sub-2", "rg-2", "ws-dev", make_current=False)
    rc = main(["use", "dev"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "switched current cluster to 'dev'" in out
    assert load_config()["current"] == "dev"


def test_use_unknown_profile_returns_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    rc = main(["use", "ghost"])
    out = capsys.readouterr().out
    assert rc == -1
    assert "no cluster profile named 'ghost'" in out
