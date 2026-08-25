"""Causal gates for provider-specific LLM candidate evaluation."""

from pathlib import Path

import pytest

import scripts.run_llm_extraction_candidates as candidate_runner
from scripts.run_llm_extraction_candidates import (
    CONSUMED_MODEL_SELECTION_PROMPT_VERSIONS,
    DEFAULT_AUDIT,
    run,
)


def test_consumed_candidates_cannot_reuse_model_selection_batches(
):
    assert {
        "structured-extraction-prompt-v18",
        "structured-extraction-prompt-v19",
        "structured-extraction-prompt-v20",
        "structured-extraction-prompt-v21",
        "structured-extraction-prompt-v22",
    }.issubset(CONSUMED_MODEL_SELECTION_PROMPT_VERSIONS)


def test_current_consumed_candidate_cannot_be_reused(tmp_path: Path):
    # Keep the duplicate-request gate covered independently of the currently
    # frozen batch state.
    original = candidate_runner.CONSUMED_MODEL_SELECTION_PROMPT_VERSIONS
    candidate_runner.CONSUMED_MODEL_SELECTION_PROMPT_VERSIONS = (
        original | {"structured-extraction-prompt-v22"}
    )
    with pytest.raises(
        PermissionError,
        match="already consumed|prompt-specific blind gold batch",
    ):
        run(
            audit_path=DEFAULT_AUDIT,
            output_path=tmp_path / "must-not-exist.jsonl",
            roles={"model_selection_validation"},
            model="deepseek-v4-flash",
            limit=0,
            dry_run=True,
        )
    candidate_runner.CONSUMED_MODEL_SELECTION_PROMPT_VERSIONS = original


def test_unconsumed_candidate_requires_fresh_prompt_specific_gold_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        candidate_runner,
        "CONSUMED_MODEL_SELECTION_PROMPT_VERSIONS",
        CONSUMED_MODEL_SELECTION_PROMPT_VERSIONS - {"structured-extraction-prompt-v22"},
    )
    monkeypatch.setitem(
        candidate_runner.MODEL_SELECTION_GOLD_LOCK_BY_PROMPT_VERSION,
        "structured-extraction-prompt-v22",
        tmp_path / "missing.lock.json",
    )
    with pytest.raises(PermissionError, match="prompt-specific blind gold batch"):
        run(
            audit_path=DEFAULT_AUDIT,
            output_path=tmp_path / "must-not-exist.jsonl",
            roles={"model_selection_validation"},
            model="deepseek-v4-flash",
            limit=0,
            dry_run=True,
        )


def test_extraction_holdout_remains_locked_for_deepseek_candidate(tmp_path: Path):
    with pytest.raises(PermissionError, match="holdout is locked"):
        run(
            audit_path=DEFAULT_AUDIT,
            output_path=tmp_path / "must-not-exist.jsonl",
            roles={"extraction_holdout"},
            model="deepseek-v4-flash",
            limit=0,
            dry_run=True,
        )
