"""Trainer-agnostic exporters for trajectory datasets (GH #66).

Every exporter consumes an iterable of trajectories (streamed) and writes rows::

    from agentbox.export import export
    export(iter_trajectories("runs/r1"), "chat", "sft.jsonl", successful_only=True)

Formats: ``chat`` (SFT messages+tools), ``dpo`` (chosen/rejected pairs),
``grpo`` (groups with advantages), ``tokens`` (prompt/response ids + masks),
``hf`` (parquet via HF datasets), ``sharegpt``, ``art``.
"""

from agentbox.export.core import (
    ExportFilter,
    ExportStats,
    available_formats,
    export,
    register_exporter,
)

__all__ = ["ExportFilter", "ExportStats", "available_formats", "export", "register_exporter"]
