"""The chat module's pure rules (S14-B3, B6, B7, B10).

What can be pinned without a model or a database: the compaction boundary,
the selection gate, the simulated-history framing (verbatim contract), and
the wire model that must never carry `state`.
"""

from __future__ import annotations

import pytest

from app.chat import (
    FOLD,
    SIMULATED_HISTORY_RULES,
    WINDOW,
    extend_system_prompt,
    fold_plan,
    selection_refusal,
)
from app.routers.chat import ChatMessageOut, ReplyOut, SessionDetailOut, SessionOut

# --- Compaction arithmetic (S14-B7) -----------------------------------------


def test_window_and_fold_are_the_locked_numbers():
    assert WINDOW == 40
    assert FOLD == 20


def test_no_fold_while_the_window_fits():
    assert fold_plan(0, 0) is None
    assert fold_plan(40, 0) is None  # exactly full is not overflowing


def test_first_fold_takes_seqs_1_to_20():
    # 41 stored messages, nothing folded: the oldest 20 go.
    assert fold_plan(41, 0) == (1, 20)


def test_second_fold_continues_from_the_fold_point():
    # 61 stored, 20 folded: live window is 41 → fold 21..40.
    assert fold_plan(61, 20) == (21, 40)
    # 60 stored, 20 folded: live window is exactly 40 → fits.
    assert fold_plan(60, 20) is None


def test_the_ac4_threshold_is_the_first_fold():
    # "A conversation driven past 60 messages triggers compaction" (AC4):
    # the FIRST fold happens as the 41st message is about to be written, so
    # by 60 messages exactly one fold has run and the second is due the
    # moment the 61st is sent.
    folds = 0
    upto = 0
    for total in range(61):
        plan = fold_plan(total, upto)
        if plan:
            folds += 1
            upto = plan[1]
    assert folds == 1
    assert upto == 20
    assert fold_plan(61, upto) == (21, 40)


# --- Selection gate (S14-B3) ------------------------------------------------


def test_complete_and_candidate_proceeds():
    assert selection_refusal("complete", is_candidate=True, already_selected=False) is None


@pytest.mark.parametrize("status", ["matching", "matched", "simulating", "failed", "no_candidates"])
def test_not_complete_is_state_not_failure(status):
    http, code, message = selection_refusal(status, is_candidate=True, already_selected=False)
    assert http == 409
    assert code == "analysis_not_complete"
    assert message


def test_already_selected_wins_over_everything():
    http, code, _ = selection_refusal("complete", is_candidate=False, already_selected=True)
    assert (http, code) == (409, "already_selected")


def test_stranger_is_not_a_candidate():
    http, code, _ = selection_refusal("complete", is_candidate=False, already_selected=False)
    assert (http, code) == (404, "not_a_candidate")


# --- The framing, verbatim (S14-B6) -----------------------------------------


def test_prompt_extension_carries_both_parts_and_names_the_simulation():
    prompt = extend_system_prompt(
        "You are Dan.", user_name="Alice",
        digest="At the Vintage Bike Exhibition: you got on.", summary=None,
    )
    assert prompt.startswith("You are Dan.")
    assert "At the Vintage Bike Exhibition: you got on." in prompt
    # The contract sentences, not a paraphrase.
    assert "Alice was NOT there" in prompt
    assert 'Call it "our simulated date"' in prompt
    assert "never say \"remember when\"" in prompt
    assert "Do not invent any detail" in prompt
    assert "EARLIER IN THIS CHAT" not in prompt


def test_prompt_extension_adds_the_summary_only_when_there_is_one():
    prompt = extend_system_prompt(
        "You are Dan.", user_name="Alice", digest="d", summary="They talked about bikes.",
    )
    assert "EARLIER IN THIS CHAT" in prompt
    assert "They talked about bikes." in prompt


def test_rules_text_only_interpolates_name_and_digest():
    # Anything else in braces would be a KeyError at call time — pin it.
    SIMULATED_HISTORY_RULES.format(name="X", digest="Y")


def test_empty_digest_is_stated_not_hidden():
    prompt = extend_system_prompt("P", user_name="A", digest="   ", summary=None)
    assert "Nothing was recorded about it." in prompt


# --- The wire never carries `state` (S14-B5) --------------------------------


@pytest.mark.parametrize("model", [ChatMessageOut, ReplyOut, SessionOut, SessionDetailOut])
def test_no_chat_payload_has_a_state_field(model):
    assert "state" not in model.model_fields
    # And nothing nested either: the message model is the only place it could hide.
    assert "state" not in ChatMessageOut.model_fields
