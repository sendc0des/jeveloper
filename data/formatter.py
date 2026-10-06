"""
Prompt and token formatters for single-pass decision tasks.
Formats State, Question, and Options into tokenized sequences and tracks
the exact token indices of [OPT] marker positions for dynamic choice pooling.
"""
from typing import List, Dict, Any, Tuple, Union, Optional
from client.schemas import OptionItem, ChoiceRequest, ScoreRequest, NoulRequest


class DecisionFormatter:
    """
    Standardized formatter for Jev decision tasks.
    """
    OPTION_MARKER = "[OPT]"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.OPTION_MARKER not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.OPTION_MARKER]})
        self.option_marker_id = self.tokenizer.convert_tokens_to_ids(self.OPTION_MARKER)

    def format_choice(
        self,
        state: str,
        question: str,
        options: List[Union[str, OptionItem, dict]],
        max_length: int = 2048
    ) -> Dict[str, Any]:
        """
        Formats a Choice request into a single-pass tokenized sequence.
        Returns:
            - input_ids: 1D Tensor of token IDs
            - attention_mask: 1D Tensor
            - marker_indices: List[int] containing token position of each option marker
            - option_ids: List[str] ordered identifiers for each candidate
        """
        parsed_options = [OptionItem.from_input(opt) for opt in options]
        option_ids = [opt.id for opt in parsed_options]

        # Construct textual representation
        # [CLS] State: <state> [SEP] Question: <question> [SEP] Options: [OPT] opt1 [OPT] opt2 ... [SEP]
        prompt_prefix = f"State: {state.strip()}\nQuestion: {question.strip()}\nOptions:\n"
        
        options_text = ""
        for opt in parsed_options:
            if opt.description:
                options_text += f"{self.OPTION_MARKER} {opt.id}: {opt.description.strip()} "
            else:
                options_text += f"{self.OPTION_MARKER} {opt.id} "

        full_text = prompt_prefix + options_text.strip()
        encoded = self.tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]

        # Locate marker positions in token sequence
        marker_positions = (input_ids == self.option_marker_id).nonzero(as_tuple=True)[0].tolist()

        # Sanity check: Ensure we found one marker per option
        if len(marker_positions) != len(parsed_options):
            # If truncation occurred or tokenizer merged tokens, reconstruct fallback indices
            # Truncate options to match found markers
            option_ids = option_ids[:len(marker_positions)]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "marker_indices": marker_positions,
            "option_ids": option_ids,
            "formatted_text": full_text
        }

    def format_score(
        self,
        state: str,
        question: str,
        max_length: int = 2048
    ) -> Dict[str, Any]:
        """
        Formats a Score evaluation request.
        """
        full_text = f"State: {state.strip()}\nEvaluate: {question.strip()}"
        encoded = self.tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "formatted_text": full_text
        }

    def format_noul(
        self,
        state: str,
        hypothesis: str,
        max_length: int = 2048
    ) -> Dict[str, Any]:
        """
        Formats a Noul binary hypothesis check.
        """
        full_text = f"State: {state.strip()}\nHypothesis: {hypothesis.strip()}"
        encoded = self.tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "formatted_text": full_text
        }
