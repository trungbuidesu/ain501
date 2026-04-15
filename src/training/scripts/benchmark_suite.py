"""Week 6 unified ONNX benchmark: same machine, registry-driven, MD + JSON report."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from src.utils.onnx_benchmark import (
    benchmark_onnx,
    feeds_from_input_specs,
    get_nested_value,
    load_json_metric,
    onnxruntime_version,
)

DEFAULT_CONFIG = Path("configs/benchmark/week6.yaml")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def load_registry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("registry must be a JSON object")
    return payload


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(root: Path, maybe: str | None) -> Path | None:
    if not maybe:
        return None
    p = Path(maybe)
    if not p.is_absolute():
        p = root / p
    return p


def run_benchmark_on_artifact(
    root: Path,
    meta: dict[str, Any] | None,
    *,
    warmup: int,
    iterations: int,
    providers: list[str],
) -> dict[str, Any] | None:
    if not meta:
        return None
    path = resolve_path(root, str(meta.get("path", "")))
    if path is None or not path.is_file():
        return {"skipped": True, "reason": "artifact_missing", "path": str(meta.get("path"))}
    specs = meta.get("input_specs")
    if not isinstance(specs, dict):
        return {"skipped": True, "reason": "no_input_specs", "path": str(path)}
    feeds = feeds_from_input_specs(specs)
    try:
        stats = benchmark_onnx(
            path,
            feeds,
            warmup=warmup,
            iterations=iterations,
            providers=providers,
        )
    except Exception as exc:  # pragma: no cover - runtime ORT
        return {"skipped": True, "reason": f"benchmark_error:{exc}", "path": str(path)}
    stats["path"] = str(path)
    stats["skipped"] = False
    return stats


def normalize_latency(
    raw_ms: float,
    unit: str,
    normalization: dict[str, Any] | None,
) -> float:
    if not normalization:
        return raw_ms
    div = float(normalization.get("divide_batch_by", 1))
    if div != 1.0:
        return raw_ms / div
    return raw_ms


def build_row(
    root: Path,
    model: dict[str, Any],
    *,
    warmup: int,
    iterations: int,
    providers: list[str],
) -> dict[str, Any]:
    mid = str(model.get("id", "unknown"))
    primary = model.get("primary_metric") or {}
    metric_name = str(primary.get("name", "metric"))
    higher = bool(primary.get("higher_is_better", True))

    metrics_block = model.get("metrics_json") or {}
    fp32_mpath = get_nested_value(metrics_block, ["fp32", "path"])
    fp32_mkeys = get_nested_value(metrics_block, ["fp32", "keys"])
    int8_mpath = get_nested_value(metrics_block, ["int8", "path"])
    int8_mkeys = get_nested_value(metrics_block, ["int8", "keys"])

    fp32_metric: float | None = None
    int8_metric: float | None = None
    if isinstance(fp32_mpath, str) and isinstance(fp32_mkeys, list):
        fp32_metric = load_json_metric(resolve_path(root, fp32_mpath) or Path(), [str(k) for k in fp32_mkeys])
    if isinstance(int8_mpath, str) and isinstance(int8_mkeys, list):
        int8_metric = load_json_metric(resolve_path(root, int8_mpath) or Path(), [str(k) for k in int8_mkeys])

    artifacts = model.get("artifacts") or {}
    fp32_art = artifacts.get("fp32_onnx") if isinstance(artifacts.get("fp32_onnx"), dict) else None
    int8_art = artifacts.get("int8_onnx") if isinstance(artifacts.get("int8_onnx"), dict) else None

    fp32_bench = run_benchmark_on_artifact(
        root, fp32_art, warmup=warmup, iterations=iterations, providers=providers
    )
    int8_bench = run_benchmark_on_artifact(
        root, int8_art, warmup=warmup, iterations=iterations, providers=providers
    )

    unit = str(model.get("latency_unit", "ms"))
    norm = model.get("latency_normalization")
    if isinstance(norm, dict):
        norm = dict(norm)
    else:
        norm = {}

    fp32_lat: float | None = None
    int8_lat: float | None = None
    if fp32_bench and not fp32_bench.get("skipped"):
        fp32_lat = normalize_latency(float(fp32_bench["latency_ms_p50"]), unit, norm)
    if int8_bench and not int8_bench.get("skipped"):
        int8_lat = normalize_latency(float(int8_bench["latency_ms_p50"]), unit, norm)

    drop: float | None = None
    if fp32_metric is not None and int8_metric is not None:
        if higher:
            drop = fp32_metric - int8_metric
        else:
            drop = int8_metric - fp32_metric

    speedup: float | None = None
    if fp32_lat is not None and int8_lat is not None and int8_lat > 0:
        speedup = fp32_lat / int8_lat

    ram_mb: float | None = None
    if int8_bench and not int8_bench.get("skipped"):
        ram_mb = float(int8_bench.get("ram_mb_delta", float("nan")))
    elif fp32_bench and not fp32_bench.get("skipped"):
        ram_mb = float(fp32_bench.get("ram_mb_delta", float("nan")))

    return {
        "model_id": mid,
        "metric_name": metric_name,
        "fp32_metric": fp32_metric,
        "int8_metric": int8_metric,
        "metric_drop": drop,
        "higher_is_better": higher,
        "fp32_latency_ms": fp32_lat,
        "int8_latency_ms": int8_lat,
        "speedup": speedup,
        "ram_mb_delta": ram_mb,
        "latency_unit": unit,
        "benchmark_fp32": fp32_bench,
        "benchmark_int8": int8_bench,
    }


def rubric_decision(
    row: dict[str, Any],
    model: dict[str, Any],
    rubric: dict[str, Any],
) -> dict[str, Any]:
    max_drop = float(rubric.get("max_absolute_metric_drop", 0.05))
    expected = model.get("expected_latency_ms") or {}
    budget = float(expected.get("edge_cpu", float("inf")))

    drop = row.get("metric_drop")
    int8_lat = row.get("int8_latency_ms")
    reasons: list[str] = []

    metric_ok = True
    if drop is not None and abs(drop) > max_drop:
        metric_ok = False
        reasons.append(f"metric_drop_abs>{max_drop}")

    latency_ok = True
    if int8_lat is None:
        latency_ok = True
    elif int8_lat <= budget:
        pass
    else:
        latency_ok = False
        reasons.append(f"int8_latency>{budget}ms_budget")

    fp32_lat = row.get("fp32_latency_ms")
    chosen = "int8_onnx"
    if fp32_lat is None and int8_lat is None:
        chosen = "no_onnx_benchmark"
    elif int8_lat is None and fp32_lat is not None:
        chosen = "fp32_onnx_only"
    elif fp32_lat is None and int8_lat is not None:
        chosen = "int8_onnx_only"
    elif not latency_ok and fp32_lat is not None:
        chosen = "fp32_onnx_preferred"

    return {
        "chosen_artifact": chosen,
        "metric_ok": metric_ok,
        "latency_ok": latency_ok,
        "reasons": reasons,
    }


def render_markdown(
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    rubric_summary: list[dict[str, Any]],
) -> str:
    lines = [
        "# Week 6 benchmark report",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(report.get("environment", {}), indent=2),
        "```",
        "",
        "## Comparison table",
        "",
        "| Model | Metric | FP32 value | INT8 value | Drop | FP32 latency (ms) | INT8 latency (ms) | Speedup | RAM Δ (MB) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row, decision in zip(rows, rubric_summary, strict=True):
        def fmt_float(v: Any) -> str:
            if v is None:
                return "N/A"
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)

        lines.append(
            "| {mid} | {mname} | {fpv} | {ipv} | {drop} | {flt} | {ilt} | {su} | {ram} |".format(
                mid=row["model_id"],
                mname=row["metric_name"],
                fpv=fmt_float(row["fp32_metric"]),
                ipv=fmt_float(row["int8_metric"]),
                drop=fmt_float(row["metric_drop"]),
                flt=fmt_float(row["fp32_latency_ms"]),
                ilt=fmt_float(row["int8_latency_ms"]),
                su=fmt_float(row["speedup"]),
                ram=fmt_float(row["ram_mb_delta"]),
            )
        )
    lines.extend(
        [
            "",
            "## Selection rubric (automated hints)",
            "",
            "| Model | Chosen | Metric OK | Latency OK | Notes |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row, decision in zip(rows, rubric_summary, strict=True):
        notes = "; ".join(decision.get("reasons", [])) or "—"
        lines.append(
            "| {mid} | {ch} | {mo} | {lo} | {n} |".format(
                mid=row["model_id"],
                ch=decision.get("chosen_artifact", ""),
                mo=decision.get("metric_ok"),
                lo=decision.get("latency_ok"),
                n=notes,
            )
        )
    lines.extend(
        [
            "",
            "Drop convention: `FP32_metric - INT8_metric` when higher is better; inverted when lower is better.",
            "Latency columns use normalized units from `latency_unit` (YOLO divides by static batch when configured).",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", action="append", dest="models", help="Filter registry model id (repeatable)")
    parser.add_argument("--skip-benchmark", action="store_true", help="Only aggregate metrics, no ORT timing")
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    root = repo_root()
    reg_path = args.registry or resolve_path(root, str(cfg.get("registry_path", "models/registry.json")))
    if reg_path is None:
        print("registry path missing", file=sys.stderr)
        return 2
    out_dir = args.output_dir or resolve_path(root, str(cfg.get("output_dir", "reports/week6")))
    if out_dir is None:
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = load_registry(reg_path)
    models = registry.get("models", [])
    if not isinstance(models, list):
        print("registry.models must be a list", file=sys.stderr)
        return 2

    warmup = int(cfg.get("warmup", 10))
    iterations = int(cfg.get("iterations", 50))
    providers = cfg.get("providers") or ["CPUExecutionProvider"]
    if not isinstance(providers, list):
        providers = ["CPUExecutionProvider"]
    providers = [str(p) for p in providers]

    filt = set(args.models) if args.models else None
    rows: list[dict[str, Any]] = []
    rubric_summary: list[dict[str, Any]] = []

    for model in models:
        if not isinstance(model, dict):
            continue
        mid = str(model.get("id", ""))
        if filt and mid not in filt:
            continue
        if args.skip_benchmark:
            stub: dict[str, Any] = {
                "model_id": mid,
                "metric_name": str((model.get("primary_metric") or {}).get("name", "")),
                "fp32_metric": None,
                "int8_metric": None,
                "metric_drop": None,
                "higher_is_better": True,
                "fp32_latency_ms": None,
                "int8_latency_ms": None,
                "speedup": None,
                "ram_mb_delta": None,
                "latency_unit": str(model.get("latency_unit", "")),
                "benchmark_fp32": {"skipped": True},
                "benchmark_int8": {"skipped": True},
            }
            mb = model.get("metrics_json") or {}
            fp = mb.get("fp32") if isinstance(mb.get("fp32"), dict) else None
            ip = mb.get("int8") if isinstance(mb.get("int8"), dict) else None
            if fp and isinstance(fp.get("keys"), list):
                p = resolve_path(root, str(fp.get("path", "")))
                stub["fp32_metric"] = load_json_metric(p or Path(), [str(k) for k in fp["keys"]])
            if ip and isinstance(ip.get("keys"), list):
                p = resolve_path(root, str(ip.get("path", "")))
                stub["int8_metric"] = load_json_metric(p or Path(), [str(k) for k in ip["keys"]])
            pm = model.get("primary_metric") or {}
            stub["metric_name"] = str(pm.get("name", "metric"))
            stub["higher_is_better"] = bool(pm.get("higher_is_better", True))
            higher = stub["higher_is_better"]
            if stub["fp32_metric"] is not None and stub["int8_metric"] is not None:
                stub["metric_drop"] = (
                    stub["fp32_metric"] - stub["int8_metric"]
                    if higher
                    else stub["int8_metric"] - stub["fp32_metric"]
                )
            rows.append(stub)
            rubric_summary.append(rubric_decision(stub, model, cfg.get("rubric") or {}))
            continue

        row = build_row(
            root,
            model,
            warmup=warmup,
            iterations=iterations,
            providers=providers,
        )
        rows.append(row)
        rubric_summary.append(rubric_decision(row, model, cfg.get("rubric") or {}))

    report = {
        "generated_at_unix": time.time(),
        "registry_path": str(reg_path),
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "onnxruntime": onnxruntime_version(),
            "providers_requested": providers,
        },
        "config_path": str(args.config),
        "rubric": cfg.get("rubric", {}),
        "rows": rows,
        "selection": rubric_summary,
    }

    json_path = out_dir / "benchmark_report.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path = out_dir / "benchmark_report.md"
    md_path.write_text(render_markdown(report, rows, rubric_summary), encoding="utf-8")
    print(json.dumps({"wrote_json": str(json_path), "wrote_md": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
