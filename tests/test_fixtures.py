"""Each part runs and is tested alone, on its own small made-up fixture: offline, in memory (Nexus)."""
import subprocess
import sys
from pathlib import Path

from starter_agent import fixtures


def test_recall_alone_finds_the_one_session_about_the_cat():
    got = fixtures.recall_alone()
    assert [sid for sid, _ in got.sessions] == [3] and "Biscuit" in got.text


def test_bdi_alone_picks_the_top_desire_and_keeps_only_checked_changes():
    got = fixtures.bdi_alone()
    assert got["intention"] == "Replicate figure 3"
    assert got["beliefs"] == ["The mice are weighed on Mondays"]
    assert len(got["rejected"]) == 2 and "[99]" in got["rejected"][0] and "1.5" in got["rejected"][1]


def test_curiosity_alone_queues_only_the_question_it_can_quote():
    got = fixtures.curiosity_alone()
    assert got["queued"] == [("Why does the loss spike at step 400?", "not sure why the loss spikes", 2)]
    assert len(got["rejected"]) == 2


def test_each_part_runs_alone_from_the_command_line():
    out = subprocess.run([sys.executable, "-m", "starter_agent.fixtures"], capture_output=True, text=True,
                         cwd=Path(__file__).parent.parent)
    assert out.returncode == 0, out.stderr
    assert [line for line in out.stdout.splitlines() if line.startswith("---")] == [
        "--- recall, alone, on made-up data ---", "--- bdi, alone, on made-up data ---",
        "--- curiosity, alone, on made-up data ---"]
    assert "(RRF of BM25 and the hash-256 embedder)" in out.stdout
