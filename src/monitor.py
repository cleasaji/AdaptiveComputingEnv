"""
Ties the model and detector together into the "continuously learns"
loop described in the brief: each session is scored against the current
model, and only reinforces the model (updates what "normal" means) if it
wasn't itself flagged as anomalous -- otherwise a successful attack
session would train the model to consider the attack's own behavior
normal, defeating the point.
"""

from dataclasses import dataclass
from typing import List

from behavior_model import BehaviorModel
from anomaly_detector import score_sequence, session_is_anomalous, StepScore


@dataclass(frozen=True)
class SessionResult:
    apps: List[str]
    scores: List[StepScore]
    anomalous: bool
    reinforced_model: bool  # whether this session was folded into the model


class AdaptiveMonitor:
    def __init__(self, model: BehaviorModel = None, threshold: float = 0.7, min_deviations: int = 1):
        self.model = model or BehaviorModel()
        self.threshold = threshold
        self.min_deviations = min_deviations
        self.session_log: List[SessionResult] = []

    def train(self, sessions: List[List[str]]) -> None:
        """Bulk-learn an initial baseline from known-good historical sessions."""
        for session in sessions:
            self.model.learn_sequence(session)

    def observe_session(self, apps: List[str]) -> SessionResult:
        scores = score_sequence(self.model, apps, threshold=self.threshold)
        anomalous = session_is_anomalous(scores, min_deviations=self.min_deviations)

        reinforced = False
        if not anomalous:
            self.model.learn_sequence(apps)
            reinforced = True

        result = SessionResult(apps=apps, scores=scores, anomalous=anomalous, reinforced_model=reinforced)
        self.session_log.append(result)
        return result
