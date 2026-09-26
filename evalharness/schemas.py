"""Pydantic models for the entire eval pipeline.

Every LLM judge response is parsed into these — we never hand-parse raw text.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class FailureMode(str, Enum):
    """Categories a failed test case can fall into."""

    NONE = "none"
    RETRIEVAL_MISS = "retrieval_miss"      # right chunk never retrieved
    HALLUCINATION = "hallucination"        # claim not grounded in context
    INCOMPLETE = "incomplete"              # partially correct, missing detail
    IRRELEVANT = "irrelevant"              # answered a different question
    REFUSAL = "refusal"                    # model declined when it shouldn't


class TestCase(BaseModel):
    """One entry from the golden dataset."""

    id: str
    question: str
    expected_answer: str
    # doc ids that SHOULD be retrieved for this question
    relevant_doc_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    doc_id: str
    text: str
    score: float
    rank: int  # 1-indexed


class PipelineOutput(BaseModel):
    """What the system-under-test produced for one question."""

    test_case_id: str
    answer: str
    retrieved: list[RetrievedChunk]
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


class JudgeVerdict(BaseModel):
    """Forced-JSON output from the LLM judge. Scores are 1-5."""

    faithfulness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    failure_mode: FailureMode = FailureMode.NONE
    reasoning: str

    @field_validator("reasoning")
    @classmethod
    def _trim(cls, v: str) -> str:
        return v.strip()[:1000]


class RetrievalMetrics(BaseModel):
    hit: bool                # was any relevant doc in top-k
    reciprocal_rank: float   # 1/rank of first relevant doc, else 0
    precision_at_k: float


class CaseResult(BaseModel):
    """Everything we know about one test case in one run."""

    test_case_id: str
    question: str
    answer: str
    retrieval: RetrievalMetrics
    verdict: Optional[JudgeVerdict] = None
    latency_ms: float
    total_tokens: int = 0

    @property
    def passed(self) -> bool:
        if self.verdict is None:
            return self.retrieval.hit
        return (
            self.retrieval.hit
            and self.verdict.faithfulness >= 4
            and self.verdict.relevance >= 4
        )


class RunSummary(BaseModel):
    """Aggregate metrics for a full eval run — this is what CI compares."""

    run_id: str
    config_name: str
    started_at: datetime
    n_cases: int

    hit_rate: float
    mrr: float
    precision_at_k: float

    avg_faithfulness: float = 0.0
    avg_relevance: float = 0.0
    avg_completeness: float = 0.0
    pass_rate: float = 0.0

    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    total_tokens: int = 0

    def regression_fields(self) -> dict[str, float]:
        """Metrics CI guards against. Higher is better for all of these."""
        return {
            "hit_rate": self.hit_rate,
            "mrr": self.mrr,
            "avg_faithfulness": self.avg_faithfulness,
            "avg_relevance": self.avg_relevance,
            "pass_rate": self.pass_rate,
        }