"""Compare two benchmark reports."""

from __future__ import annotations

from agentbox.benchmark.schema import BenchmarkReport, ComparisonResult


def compare_reports(
    a: BenchmarkReport,
    b: BenchmarkReport,
    *,
    require_hash_match: bool = False,
) -> ComparisonResult:
    warnings: list[str] = []
    if a.suite_id != b.suite_id:
        warnings.append(f"suite_id differs: {a.suite_id!r} vs {b.suite_id!r}")
    hash_match = (a.suite_content_hash or "") == (b.suite_content_hash or "")
    if not hash_match:
        warnings.append(
            f"content_hash differs: {a.suite_content_hash!r} vs {b.suite_content_hash!r}"
        )
    if require_hash_match and not hash_match:
        warnings.append("require_hash_match=True: comparison may be unfair")

    a_models = {m.model_id: m for m in a.models}
    b_models = {m.model_id: m for m in b.models}
    deltas: list[dict] = []

    for mid in sorted(set(a_models) | set(b_models)):
        ma, mb = a_models.get(mid), b_models.get(mid)
        if ma is None or mb is None:
            warnings.append(f"model {mid!r} missing in one report")
            continue
        deltas.append(
            {
                "model_id": mid,
                "success_rate_a": ma.aggregate.success_rate,
                "success_rate_b": mb.aggregate.success_rate,
                "delta_success_rate": mb.aggregate.success_rate - ma.aggregate.success_rate,
                "mean_steps_a": ma.aggregate.mean_steps,
                "mean_steps_b": mb.aggregate.mean_steps,
                "delta_mean_steps": mb.aggregate.mean_steps - ma.aggregate.mean_steps,
            }
        )

    return ComparisonResult(
        suite_id=a.suite_id,
        hash_match=hash_match,
        warnings=warnings,
        deltas=deltas,
    )
