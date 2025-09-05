# biometria/domain/value_objects.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Thresholds:
    similarity: float = 95.0
    live: float = 0.90
    luxand: float = 0.85

@dataclass(frozen=True)
class EvaluationResult:
    is_live: bool
    is_match: bool
    liveness_score: float       # 0..1
    similarity: float           # 0..100
    evaluation_pct: float       # 0..100
    message: str
