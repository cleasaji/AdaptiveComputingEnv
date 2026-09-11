"""
Scores an observed sequence of app/process transitions against a learned
BehaviorModel and flags deviations. Deliberately combines two signals
(transition improbability + process novelty) because either alone is too
noisy: a rare-but-legitimate transition happens sometimes, and a brand
new (but harmless) app happens too -- it's the *combination* of an
unlikely transition landing on an unfamiliar process that is the
strongest signal, matching the "Chrome -> PowerShell -> unknown process"
example.
"""

from dataclasses import dataclass
from typing import List

from behavior_model import BehaviorModel


@dataclass(frozen=True)
class StepScore:
    from_app: str
    to_app: str
    transition_probability: float
    process_familiarity: float
    deviation_score: float   # 0 (fully normal) to 1 (fully novel)
    is_deviation: bool


def _max_relative_probability(model: BehaviorModel, from_app: str) -> float:
    """Highest smoothed transition probability among the from_app's own
    observed destinations -- the benchmark 'most normal thing to do next'
    that a candidate transition's probability is compared against."""
    seen_from_count = model.from_totals.get(from_app, 0)
    if seen_from_count == 0:
        return 0.0
    destinations = {to for (f, to) in model.transition_counts if f == from_app}
    return max((model.transition_probability(from_app, to) for to in destinations), default=0.0)


def score_step(model: BehaviorModel, from_app: str, to_app: str, threshold: float) -> StepScore:
    prob = model.transition_probability(from_app, to_app)
    familiarity = model.process_familiarity(to_app)

    # Deviation score blends "how unlikely was this transition RELATIVE TO
    # what's normally done from from_app" with "how unfamiliar is the
    # destination process" -- comparing against the from-state's own best
    # transition (rather than an absolute probability cutoff) is what
    # makes this personalized per machine instead of a fixed threshold.
    max_prob = _max_relative_probability(model, from_app)
    transition_deviation = 1.0 if max_prob == 0 else max(0.0, 1.0 - prob / max_prob)
    process_deviation = 1.0 - familiarity
    deviation_score = round(0.5 * transition_deviation + 0.5 * process_deviation, 3)

    return StepScore(
        from_app=from_app,
        to_app=to_app,
        transition_probability=round(prob, 4),
        process_familiarity=round(familiarity, 3),
        deviation_score=deviation_score,
        is_deviation=deviation_score >= threshold,
    )


def score_sequence(model: BehaviorModel, apps: List[str], threshold: float = 0.7) -> List[StepScore]:
    return [score_step(model, a, b, threshold) for a, b in zip(apps, apps[1:])]


def session_is_anomalous(scores: List[StepScore], min_deviations: int = 1) -> bool:
    """A session is flagged if enough of its individual steps deviate --
    a single borderline step shouldn't page anyone, but a run of them should."""
    return sum(1 for s in scores if s.is_deviation) >= min_deviations
