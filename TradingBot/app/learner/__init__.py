"""
Trade Introspection Learning System.

This module implements a self-learning system that analyzes each trade
with full hindsight, asking:
- Should I have entered earlier/later/not at all?
- Was my SL/TP optimal?
- What happened after I exited?

And learns by gradually adjusting weights based on accumulated evidence.
"""

from .introspector import TradeIntrospector
from .counterfactual import CounterfactualSimulator
from .weights import AdaptiveWeights
from .lessons import Lesson, LessonStore
from .integration import LearnerIntegration, get_learner, is_learner_enabled

__all__ = [
    "TradeIntrospector",
    "CounterfactualSimulator",
    "AdaptiveWeights",
    "Lesson",
    "LessonStore",
    "LearnerIntegration",
    "get_learner",
    "is_learner_enabled",
]
