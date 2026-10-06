"""
Unit tests for DecisionFormatter and dynamic option marker tracking.
"""
import pytest
from transformers import AutoTokenizer
from data.formatter import DecisionFormatter
from client.schemas import OptionItem


def test_decision_formatter():
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    formatter = DecisionFormatter(tokenizer)

    # 1. Choice formatting
    state = "I lost my credit card during travel."
    question = "Identify customer request."
    options = [
        OptionItem(id="lost_card", description="Customer reports lost or misplaced card."),
        OptionItem(id="check_balance", description="Account balance query."),
        "stolen_card"
    ]

    res = formatter.format_choice(state, question, options)
    assert "input_ids" in res
    assert "attention_mask" in res
    assert len(res["marker_indices"]) == 3
    assert res["option_ids"] == ["lost_card", "check_balance", "stolen_card"]

    # 2. Score formatting
    score_res = formatter.format_score("Service was great.", "Rate sentiment 1 to 5")
    assert "input_ids" in score_res
    assert len(score_res["input_ids"]) > 0

    # 3. Noul formatting
    noul_res = formatter.format_noul("Server is responsive.", "System is online.")
    assert "input_ids" in noul_res
    assert len(noul_res["input_ids"]) > 0
