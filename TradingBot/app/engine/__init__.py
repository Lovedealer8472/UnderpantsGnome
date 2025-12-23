"""
Engine Module - Scanner, executor, and exit pipeline.
Contains the main trading engine components.
"""

from .exit_pipeline import ExitPipeline, ExitRequest

__all__ = [
    "ExitPipeline",
    "ExitRequest",
]

