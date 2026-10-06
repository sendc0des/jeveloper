"""
Pydantic Schemas for Jevloper Decision Engine.
Provides strict type-safety, validation, serialization, and calibrated confidence models.
"""
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field, field_validator


class OptionItem(BaseModel):
    """
    Rich representation of an option in a Choice decision.
    Allows passing both a clean identifier and an optional semantic description.
    """
    id: str = Field(..., description="Unique identifier or label for this option.")
    description: Optional[str] = Field(None, description="Optional natural language explanation of this choice.")

    @classmethod
    def from_input(cls, item: Union[str, "OptionItem", dict]) -> "OptionItem":
        if isinstance(item, str):
            return cls(id=item.strip())
        elif isinstance(item, dict):
            return cls(**item)
        elif isinstance(item, OptionItem):
            return item
        raise ValueError(f"Cannot parse choice option from {type(item)}: {item}")


class ChoiceRequest(BaseModel):
    """
    Request model for categorical Choice decisions.
    """
    state: str = Field(..., description="The state or contextual payload (text, document, ticket).")
    question: str = Field(..., description="The question or decision prompt.")
    options: List[Union[str, OptionItem]] = Field(
        ...,
        min_length=2,
        max_length=255,
        description="List of 2 to 255 candidate choices."
    )
    conformal_alpha: Optional[float] = Field(
        None,
        ge=0.01,
        le=0.50,
        description="Optional error tolerance alpha for Conformal Prediction (e.g. 0.05 for 95% certified coverage)."
    )


class ChoiceResponse(BaseModel):
    """
    Typed, calibrated response for a categorical Choice decision.
    """
    selected: str = Field(..., description="The winning choice ID.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence score [0.0, 1.0].")
    probabilities: Dict[str, float] = Field(..., description="Full probability distribution over all choices.")
    conformal_set: Optional[List[str]] = Field(
        None,
        description="Statistically certified prediction set guaranteeing (1 - alpha) coverage."
    )
    is_ambiguous: bool = Field(
        False,
        description="True if the conformal prediction set contains more than one choice."
    )
    latency_ms: float = Field(..., ge=0.0, description="Inference latency in milliseconds.")


class ScoreRequest(BaseModel):
    """
    Request model for continuous / ordinal Score evaluations.
    """
    state: str = Field(..., description="The context payload to evaluate.")
    question: str = Field(..., description="The criterion or evaluation question.")
    min_score: float = Field(1.0, description="Minimum score boundary.")
    max_score: float = Field(5.0, description="Maximum score boundary.")
    num_bins: int = Field(5, ge=2, le=20, description="Number of discrete distribution bins across [min_score, max_score].")

    @field_validator("max_score")
    @classmethod
    def validate_bounds(cls, v: float, info) -> float:
        min_v = info.data.get("min_score", 1.0)
        if v <= min_v:
            raise ValueError(f"max_score ({v}) must be greater than min_score ({min_v})")
        return v


class ScoreResponse(BaseModel):
    """
    Typed, calibrated response for ordinal / metric scoring.
    """
    score: float = Field(..., description="Expected score value along the bounded scale.")
    variance: float = Field(..., ge=0.0, description="Variance / uncertainty of the score distribution.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence that score is in highest-density bin.")
    distribution: Dict[float, float] = Field(..., description="Probability mass across discrete score bins.")
    latency_ms: float = Field(..., ge=0.0, description="Inference latency in milliseconds.")


class NoulRequest(BaseModel):
    """
    Request model for binary Noul hypothesis validation.
    """
    state: str = Field(..., description="Context or observation.")
    hypothesis: str = Field(..., description="Hypothesis or statement to test (True/False).")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Decision threshold for True.")
    conformal_alpha: Optional[float] = Field(
        None,
        ge=0.01,
        le=0.50,
        description="Optional error tolerance alpha for Conformal Prediction."
    )


class NoulResponse(BaseModel):
    """
    Typed, calibrated response for binary truth checks.
    """
    result: bool = Field(..., description="Binary outcome: True if probability >= threshold, else False.")
    probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated probability of hypothesis being True.")
    confidence: float = Field(..., ge=0.5, le=1.0, description="Certainty score = max(probability, 1 - probability).")
    conformal_set: Optional[List[bool]] = Field(
        None,
        description="Certified truth set: [True], [False], or [True, False] if indeterminate."
    )
    latency_ms: float = Field(..., ge=0.0, description="Inference latency in milliseconds.")
