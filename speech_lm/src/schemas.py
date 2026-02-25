"""
Pydantic schemas for structured validation of SLM scam detection outputs.
"""

from pydantic import BaseModel, Field


class ScamAnalysisResult(BaseModel):
    """Expected JSON structure from the scam detection model."""

    is_scam: bool
    fraud_type: str  # Impersonation | Investment | Tech Support | Normal
    acoustic_analysis: str
    semantic_analysis: str
    confidence_score: int = Field(ge=0, le=100)


def validate_result(raw: dict) -> tuple["ScamAnalysisResult | None", str | None]:
    """
    Validate parsed JSON against the expected schema.

    Returns:
        (ScamAnalysisResult, None) on success, or (None, error_message) on failure.
    """
    try:
        return ScamAnalysisResult.model_validate(raw), None
    except Exception as e:
        return None, str(e)
