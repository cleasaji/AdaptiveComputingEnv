"""
A per-machine model of "normal" computing behavior, built from two
signals:

1. Application-switch transitions (Chrome -> VS Code -> Terminal), modeled
   as a first-order Markov chain: P(next_app | current_app). A sequence
   that's individually plausible app-by-app can still be a wildly unlikely
   *transition* -- that's what a bag-of-apps model would miss and a
   transition model catches.
2. Process familiarity: how often each process name has been seen before,
   used to catch a never-before-seen process regardless of what it
   transitions from/to.

No fixed rule set is hardcoded -- "normal" is entirely defined by what
this specific instance has observed via observe_transition()/observe_process(),
which is what makes the model personalized per machine rather than a
shared signature list.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class BehaviorModel:
    transition_counts: Dict[Tuple[str, str], int] = field(default_factory=dict)
    from_totals: Dict[str, int] = field(default_factory=dict)
    process_counts: Dict[str, int] = field(default_factory=dict)
    total_observations: int = 0

    def observe_transition(self, from_app: str, to_app: str) -> None:
        key = (from_app, to_app)
        self.transition_counts[key] = self.transition_counts.get(key, 0) + 1
        self.from_totals[from_app] = self.from_totals.get(from_app, 0) + 1

    def observe_process(self, process_name: str) -> None:
        self.process_counts[process_name] = self.process_counts.get(process_name, 0) + 1
        self.total_observations += 1

    def learn_sequence(self, apps: list) -> None:
        """Convenience: feed a full session (ordered list of app/process
        names) into the model as both transitions and process sightings."""
        for name in apps:
            self.observe_process(name)
        for a, b in zip(apps, apps[1:]):
            self.observe_transition(a, b)

    def transition_probability(self, from_app: str, to_app: str, smoothing: float = 0.5) -> float:
        """
        Laplace-smoothed probability so an unseen transition gets a small
        nonzero probability rather than exactly 0 (which would make the
        anomaly score infinite/undefined) while still being clearly lower
        than any transition that's actually been observed.
        """
        seen_from_count = self.from_totals.get(from_app, 0)
        seen_pair_count = self.transition_counts.get((from_app, to_app), 0)
        vocab_size = max(1, len(self.from_totals))
        return (seen_pair_count + smoothing) / (seen_from_count + smoothing * vocab_size)

    def process_familiarity(self, process_name: str) -> float:
        """0.0 for never seen, up to 1.0 for the most frequently seen process."""
        if self.total_observations == 0:
            return 0.0
        count = self.process_counts.get(process_name, 0)
        max_count = max(self.process_counts.values()) if self.process_counts else 1
        return count / max_count if count > 0 else 0.0

    def is_known_process(self, process_name: str) -> bool:
        return process_name in self.process_counts
