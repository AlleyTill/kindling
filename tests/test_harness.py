"""tests/mutants.py, tested: it copies only the code, a hanging mutant is a row and not the end of the run, and
only a test failing counts as a kill. (Round 3: the Agent Army's test-engineer and adversarial-reviewer.)"""
import subprocess

import mutants


def test_the_snapshot_copies_the_code_and_never_a_venv_secrets_or_memory(tmp_path):
    root, frozen = tmp_path / "kindling", tmp_path / "snapshot"
    for name in ("starter_agent/agent.py", "tests/test_x.py", "pytest.ini", "README.md", ".venv/lib/torch.so",
                 ".env", "agent.json", "memory/agent.db", "memory/agent.db-journal"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("x")
    mutants.snapshot(root, frozen)
    copied = sorted(str(p.relative_to(frozen)) for p in frozen.rglob("*") if p.is_file())
    assert copied == ["README.md", "pytest.ini", "starter_agent/agent.py", "tests/test_x.py"]


def test_a_mutant_that_hangs_is_a_timeout_row(tmp_path, monkeypatch):
    def hangs(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs.get("timeout"))
    monkeypatch.setattr(mutants.subprocess, "run", hangs)
    try:
        run = mutants.run_tests(tmp_path)
    except BaseException as error:              # the whole run ending must fail by assertion
        run = error
    assert run == {"exit": "timeout", "ran": 0, "failed": []}
    assert mutants.verdict(run)[0] == "TIMEOUT"


def test_a_mutant_that_isnt_valid_python_is_not_applied(tmp_path):
    (tmp_path / "snapshot" / "pkg").mkdir(parents=True)
    (tmp_path / "snapshot" / "pkg" / "a.py").write_text("try:\n    x = 1  # one, two\nexcept OSError:\n    pass\n")
    folder, problem = mutants.mutate(tmp_path / "snapshot", 1, "pkg/a.py", [("x = 1  # one", "if True:")])
    assert folder is None and "isn't valid Python" in problem


def test_only_a_test_failing_counts_as_a_kill():
    def judge(kind, where):
        return mutants.verdict({"exit": 1, "ran": 9, "failed": [["tests/test_a.py::test_it", "call", kind, where]]})[0]
    assert judge("AssertionError", "/w/mutant_01/tests/test_a.py") == "KILLED"          # an assert in a test
    assert judge("AssertionError", "/w/mutant_01/tests/conftest.py") == "KILLED"        # the network guard
    assert judge("Failed", "/venv/_pytest/python_api.py") == "KILLED"                   # pytest.raises: DID NOT RAISE
    assert judge("AssertionError", "/w/mutant_01/starter_agent/store.py") == "INVALID"  # an assert in the package
    assert judge("KeyError", "/w/mutant_01/starter_agent/bdi.py") == "INVALID"          # a crash
    assert mutants.verdict({"exit": 0, "ran": 9, "failed": []})[0] == "SURVIVED"


def test_a_mutant_in_the_readme_is_planted_as_text(tmp_path):
    """Round 7: the README's numbers are checked against the code, so a README mutant must be planted, not refused."""
    (tmp_path / "snapshot").mkdir()
    (tmp_path / "snapshot" / "README.md").write_text("In a 4,096-token window, at most 2,457 tokens.\n")
    folder, problem = mutants.mutate(tmp_path / "snapshot", 1, "README.md", [("2,457", "3,072")])
    assert problem is None, problem
    assert (folder / "README.md").read_text() == "In a 4,096-token window, at most 3,072 tokens.\n"
