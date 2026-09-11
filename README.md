# 🧠 AdaptiveComputingEnv

A behavioral-security model that learns **how a specific machine is
normally used** — application-switch patterns and process familiarity —
and flags deviations, instead of relying on a fixed signature list.

> Research-oriented portfolio project answering: *can a computer detect
> attacks by learning the user's own normal interaction behavior, rather
> than primarily relying on predefined signatures?*

---

## The core idea

```
Normal:  Chrome -> VS Code -> Terminal -> VS Code     (seen many times)
Attack:  Chrome -> PowerShell -> unknown_process -> encrypted_archive
```

Both sequences are made of "app switches" — a bag-of-apps model can't
tell them apart. What actually differs is the *transition*: this machine
has never gone from Chrome to PowerShell to an unrecognized process, and
that specific combination — an unlikely transition landing on an
unfamiliar process — is what gets flagged, not either signal alone.

## Two signals, combined on purpose

**Transition probability** (`behavior_model.py`) — a first-order Markov
model: `P(next_app | current_app)`, Laplace-smoothed so a never-seen
transition gets a small nonzero probability instead of undefined/zero.

**Process familiarity** — how often a given process name has been seen
before, independent of what it followed. Catches an unfamiliar process
regardless of the path that led to it.

`anomaly_detector.py` scores each step by comparing its transition
probability against *that same starting app's own most common next
step* — not a fixed global threshold — which is what makes "normal"
personalized per machine rather than a shared rule.

## Continuous learning, without poisoning itself

`monitor.py`'s `AdaptiveMonitor` reinforces the model with every
non-anomalous session — "normal" keeps evolving with actual use. But a
session flagged anomalous is **not** folded back in: if it were, a
successful attack session would train the model to consider the
attacker's own behavior normal on the next pass. The test suite checks
this directly — an anomalous session leaves `transition_counts`
untouched.

## Example

```python
from monitor import AdaptiveMonitor

monitor = AdaptiveMonitor(threshold=0.7)
monitor.train([
    ["Chrome", "VSCode", "Terminal", "VSCode"],
    ["Chrome", "VSCode", "Terminal", "VSCode"],
    ["VSCode", "Terminal", "Chrome", "VSCode"],
])

result = monitor.observe_session(
    ["Chrome", "PowerShell", "unknown_process.exe", "encrypted_archive.exe"]
)
print(result.anomalous)          # True
for s in result.scores:
    print(s.from_app, "->", s.to_app, "deviation:", s.deviation_score)
```

## Why a per-machine Markov model over a fixed IDS ruleset

A signature-based IDS knows what *known* attacks look like. This asks a
different question: what does *this* machine's owner normally do, and
how far does a given session stray from that — which can catch behavior
that doesn't match any known signature, at the cost of needing a
baseline training period and being tuned to one machine's habits rather
than portable across machines.

## Tests

```bash
pip install -r requirements.txt
cd tests && python -m pytest -v
```

9 tests: transition counts accumulate correctly, an unseen transition
scores lower probability than a seen one, process familiarity is zero
for never-seen processes, normal sequences score low deviation, an
unfamiliar process scores high deviation, the brief's exact attack
sequence gets flagged, a normal session reinforces the model, an
anomalous session does *not* reinforce it, and the deviation-count
threshold is respected.

## Project layout

```
src/
  behavior_model.py     # Markov transition counts + process familiarity
  anomaly_detector.py     # per-step and per-session deviation scoring
  monitor.py                # online learning loop (reinforce only non-anomalous sessions)
tests/
  test_adaptivecomputingenv.py
```

## Honest scope

First-order Markov chains only (no keyboard/mouse timing, file-access, or
CPU/network signals modeled yet, though the architecture — observe,
score against a learned baseline, conditionally reinforce — extends to
those the same way). Small in-memory model, not tuned against a real
attack dataset; the point is the architecture for personalized
behavioral baselining, not a production EDR replacement.
