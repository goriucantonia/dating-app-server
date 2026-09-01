"""The `/simulate` gate, as a pure decision (S13-U5, S13-B1).

Step 13's retry button says "picks up where it stopped". For that sentence to
be true, the server has to let a `failed` analysis back into the pipeline —
and only when there is something to pick up. These tests pin the boundary
(§18): failed-with-candidates resumes; failed-in-matching does not; nothing
else about the gate moved.
"""

from __future__ import annotations

import pytest

from app.routers.simulation import simulate_refusal


def test_matched_proceeds():
    assert simulate_refusal("matched", has_candidates=True) is None


def test_failed_after_matching_resumes():
    # The pipeline died mid-date; the rows are checkpointed; retry resumes.
    assert simulate_refusal("failed", has_candidates=True) is None


def test_failed_in_matching_does_not_resume():
    # Nothing was matched, so there is nothing to resume from — the honest
    # answer names that rather than spinning up a pipeline with no dates.
    code, message = simulate_refusal("failed", has_candidates=False)
    assert code == "not_ready_to_simulate"
    assert "Start a new one" in message


def test_simulating_is_state_not_failure():
    code, _ = simulate_refusal("simulating", has_candidates=True)
    assert code == "simulation_in_progress"


@pytest.mark.parametrize("status", ["matching", "no_candidates", "complete"])
def test_other_states_still_refused(status):
    code, message = simulate_refusal(status, has_candidates=True)
    assert code == "not_ready_to_simulate"
    assert message  # every refusal carries a layman sentence (§26)


def test_matched_without_candidate_rows_is_still_allowed():
    # `matched` implies candidates exist; the flag only matters for `failed`.
    # Pinned so a future edit does not quietly make the flag load-bearing for
    # every state.
    assert simulate_refusal("matched", has_candidates=False) is None
