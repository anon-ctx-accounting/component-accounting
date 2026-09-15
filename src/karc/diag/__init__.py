"""Reusable, model-free curation-activation diagnostic instrument."""

from karc.diag.gates import THRESHOLDS, benchmark_gate, family_gate
from karc.diag.model import (
    CapacityCell,
    IngestionRecord,
    LeakageError,
    MechanismFamily,
    MeasurementOnly,
    NativeSignal,
    PolicyQueryView,
    QueryPoint,
    SignalKind,
    SignalSource,
)
from karc.diag.replay import audit_families

__all__ = [
    "THRESHOLDS",
    "CapacityCell",
    "IngestionRecord",
    "LeakageError",
    "MechanismFamily",
    "MeasurementOnly",
    "NativeSignal",
    "PolicyQueryView",
    "QueryPoint",
    "SignalKind",
    "SignalSource",
    "audit_families",
    "benchmark_gate",
    "family_gate",
]
