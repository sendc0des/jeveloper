"""
Jev Client API.
High-level developer interface for local decision-making:
- client.choice(state, question, options)
- client.score(state, question, min_score, max_score)
- client.noul(state, hypothesis, threshold)
"""
import time
import numpy as np
import torch
from typing import List, Dict, Optional, Union

from core.unified_model import JevModel
from data.formatter import DecisionFormatter
from client.schemas import (
    OptionItem, ChoiceRequest, ChoiceResponse,
    ScoreRequest, ScoreResponse,
    NoulRequest, NoulResponse
)
from calibration.conformal import ConformalPredictor


class JevClient:
    """
    Local System-1 Decision Client.
    Direct drop-in decision engine replacing LLMs with sub-15ms calibrated decisions.
    """
    def __init__(
        self,
        model: Optional[JevModel] = None,
        model_id: str = "answerdotai/ModernBERT-base",
        checkpoint_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if model is not None:
            self.model = model
        else:
            print(f"[*] Initializing JevModel on {self.device}...")
            self.model = JevModel(model_id=model_id, device=self.device)
            if checkpoint_dir:
                self.model.load_heads(checkpoint_dir)

        self.model.eval()
        self.formatter = DecisionFormatter(self.model.tokenizer)
        self.conformal = ConformalPredictor()

    @torch.no_grad()
    def choice(
        self,
        state: str,
        question: str,
        options: List[Union[str, OptionItem, dict]],
        conformal_alpha: Optional[float] = None
    ) -> ChoiceResponse:
        """
        Executes categorical Choice decision.
        """
        t0 = time.perf_counter()

        formatted = self.formatter.format_choice(state=state, question=question, options=options)
        input_ids = formatted["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = formatted["attention_mask"].unsqueeze(0).to(self.device)
        marker_indices = [formatted["marker_indices"]]
        option_ids = formatted["option_ids"]

        outputs = self.model.forward_choice(
            input_ids=input_ids,
            attention_mask=attention_mask,
            marker_indices=marker_indices
        )

        probs_tensor = outputs["probabilities"][0].cpu()  # (K,)
        probs_np = probs_tensor.numpy()
        probs_dict = {opt_id: float(p) for opt_id, p in zip(option_ids, probs_np)}

        best_idx = int(torch.argmax(probs_tensor).item())
        selected = option_ids[best_idx]
        confidence = float(probs_np[best_idx])

        # Conformal prediction set
        conformal_set = None
        is_ambiguous = False
        if conformal_alpha is not None or self.conformal.is_calibrated:
            alpha = conformal_alpha or 0.05
            conformal_set = self.conformal.predict_set(
                probabilities=probs_np,
                candidate_ids=option_ids,
                alpha=alpha
            )
            is_ambiguous = len(conformal_set) > 1

        latency_ms = (time.perf_counter() - t0) * 1000

        return ChoiceResponse(
            selected=selected,
            confidence=confidence,
            probabilities=probs_dict,
            conformal_set=conformal_set,
            is_ambiguous=is_ambiguous,
            latency_ms=latency_ms
        )

    @torch.no_grad()
    def score(
        self,
        state: str,
        question: str,
        min_score: float = 1.0,
        max_score: float = 5.0
    ) -> ScoreResponse:
        """
        Executes bounded continuous / ordinal Score evaluation.
        """
        t0 = time.perf_counter()

        formatted = self.formatter.format_score(state=state, question=question)
        input_ids = formatted["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = formatted["attention_mask"].unsqueeze(0).to(self.device)

        outputs = self.model.forward_score(
            input_ids=input_ids,
            attention_mask=attention_mask,
            min_score=min_score,
            max_score=max_score
        )

        score_val = float(outputs["expected_score"][0].cpu().item())
        variance_val = float(outputs["variance"][0].cpu().item())
        probs_np = outputs["probs"][0].cpu().numpy()
        bin_centers_np = outputs["bin_centers"].cpu().numpy()

        dist_dict = {float(f"{c:.2f}"): float(p) for c, p in zip(bin_centers_np, probs_np)}
        confidence = float(np.max(probs_np))

        latency_ms = (time.perf_counter() - t0) * 1000

        return ScoreResponse(
            score=score_val,
            variance=variance_val,
            confidence=confidence,
            distribution=dist_dict,
            latency_ms=latency_ms
        )

    @torch.no_grad()
    def noul(
        self,
        state: str,
        hypothesis: str,
        threshold: float = 0.5,
        conformal_alpha: Optional[float] = None
    ) -> NoulResponse:
        """
        Executes binary Noul truth verification.
        """
        t0 = time.perf_counter()

        formatted = self.formatter.format_noul(state=state, hypothesis=hypothesis)
        input_ids = formatted["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = formatted["attention_mask"].unsqueeze(0).to(self.device)

        outputs = self.model.forward_noul(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        prob_true = float(outputs["probability"][0].cpu().item())
        result = prob_true >= threshold
        confidence = float(outputs["confidence"][0].cpu().item())

        conformal_set = None
        if conformal_alpha is not None:
            c_set = []
            if prob_true >= (conformal_alpha / 2):
                c_set.append(True)
            if (1.0 - prob_true) >= (conformal_alpha / 2):
                c_set.append(False)
            conformal_set = c_set or [result]

        latency_ms = (time.perf_counter() - t0) * 1000

        return NoulResponse(
            result=result,
            probability=prob_true,
            confidence=confidence,
            conformal_set=conformal_set,
            latency_ms=latency_ms
        )
