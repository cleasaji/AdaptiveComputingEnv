import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from behavior_model import BehaviorModel
from anomaly_detector import score_step, score_sequence, session_is_anomalous
from monitor import AdaptiveMonitor


NORMAL_SESSIONS = [
    ["Chrome", "VSCode", "Terminal", "VSCode"],
    ["Chrome", "VSCode", "Terminal", "VSCode"],
    ["VSCode", "Terminal", "Chrome", "VSCode"],
    ["Chrome", "VSCode", "Terminal", "VSCode"],
]


def train_baseline():
    monitor = AdaptiveMonitor(threshold=0.7)
    monitor.train(NORMAL_SESSIONS)
    return monitor


def test_model_learns_transition_counts():
    model = BehaviorModel()
    model.learn_sequence(["Chrome", "VSCode", "Terminal"])
    assert model.transition_counts[("Chrome", "VSCode")] == 1
    assert model.transition_counts[("VSCode", "Terminal")] == 1


def test_unseen_transition_has_lower_probability_than_seen_one():
    model = BehaviorModel()
    for _ in range(10):
        model.learn_sequence(["Chrome", "VSCode"])
    seen = model.transition_probability("Chrome", "VSCode")
    unseen = model.transition_probability("Chrome", "PowerShell")
    assert seen > unseen


def test_process_familiarity_zero_for_never_seen():
    model = BehaviorModel()
    model.learn_sequence(["Chrome", "VSCode"])
    assert model.process_familiarity("PowerShell") == 0.0
    assert model.process_familiarity("Chrome") > 0.0


def test_normal_transition_scores_low_deviation():
    monitor = train_baseline()
    scores = score_sequence(monitor.model, ["Chrome", "VSCode", "Terminal"], threshold=0.7)
    assert all(s.deviation_score < 0.7 for s in scores)


def test_unfamiliar_process_scores_high_deviation():
    monitor = train_baseline()
    score = score_step(monitor.model, "Chrome", "unknown_process.exe", threshold=0.7)
    assert score.is_deviation
    assert score.process_familiarity == 0.0


def test_session_flagged_anomalous_with_unseen_process_chain():
    monitor = train_baseline()
    # Matches the brief's example: normal chain, then a sharp departure.
    result = monitor.observe_session(["Chrome", "PowerShell", "unknown_process.exe", "encrypted_archive.exe"])
    assert result.anomalous


def test_normal_session_not_flagged_and_reinforces_model():
    monitor = train_baseline()
    before = dict(monitor.model.transition_counts)
    result = monitor.observe_session(["Chrome", "VSCode", "Terminal", "VSCode"])
    assert not result.anomalous
    assert result.reinforced_model
    assert monitor.model.transition_counts != before  # counts increased


def test_anomalous_session_does_not_reinforce_model():
    monitor = train_baseline()
    before = dict(monitor.model.transition_counts)
    result = monitor.observe_session(["Chrome", "PowerShell", "unknown_process.exe"])
    assert result.anomalous
    assert not result.reinforced_model
    assert monitor.model.transition_counts == before  # unchanged -- attack didn't poison the model


def test_session_is_anomalous_respects_min_deviations_threshold():
    monitor = train_baseline()
    scores = score_sequence(monitor.model, ["Chrome", "VSCode"], threshold=0.7)
    assert session_is_anomalous(scores, min_deviations=5) is False
