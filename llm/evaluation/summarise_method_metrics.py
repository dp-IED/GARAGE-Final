#!/usr/bin/env python3
"""
Summarise overall evaluation metrics across methods.

Supports multiple result schemas currently used in this repository:
1) Unified compare schema (metrics.window / metrics.sensor / metrics.fault_type)
2) Legacy evaluator schema (metrics.window_level / metrics.sensor_level)
3) GDN retrained schema (classification_metrics / sensor_attribution_metrics)
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _method_name(payload: Dict[str, Any], path: Path) -> str:
    if isinstance(payload.get("method"), str) and payload["method"].strip():
        return payload["method"].strip()
    return path.stem


def _extract_unified(payload: Dict[str, Any]) -> Optional[Dict[str, Optional[float]]]:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return None
    if "window" not in metrics or "sensor" not in metrics:
        return None

    window = metrics.get("window", {})
    sensor = metrics.get("sensor", {})
    fault_type = metrics.get("fault_type", {})
    bert = metrics.get("bertscore", {})

    return {
        "num_windows": _safe_float(payload.get("num_windows")),
        "window_acc": _safe_float(window.get("accuracy")),
        "window_precision": _safe_float(window.get("precision")),
        "window_recall": _safe_float(window.get("recall")),
        "window_f1": _safe_float(window.get("f1")),
        "sensor_precision": _safe_float(sensor.get("precision")),
        "sensor_recall": _safe_float(sensor.get("recall")),
        "sensor_f1": _safe_float(sensor.get("f1")),
        "fault_type_acc": _safe_float(fault_type.get("accuracy")),
        "bertscore_f1": _safe_float(bert.get("f1")),
    }


def _extract_legacy(payload: Dict[str, Any]) -> Optional[Dict[str, Optional[float]]]:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return None
    if "window_level" not in metrics or "sensor_level" not in metrics:
        return None

    window = metrics.get("window_level", {})
    sensor = metrics.get("sensor_level", {})
    efficiency = metrics.get("efficiency", {})

    return {
        "num_windows": _safe_float(payload.get("num_windows")) or _safe_float(efficiency.get("num_windows")),
        "window_acc": _safe_float(window.get("window_accuracy")),
        "window_precision": _safe_float(window.get("window_precision")),
        "window_recall": _safe_float(window.get("window_recall")),
        "window_f1": _safe_float(window.get("window_f1")),
        "sensor_precision": _safe_float(sensor.get("sensor_precision")),
        "sensor_recall": _safe_float(sensor.get("sensor_recall")),
        "sensor_f1": _safe_float(sensor.get("sensor_f1")),
        "fault_type_acc": None,
        "bertscore_f1": None,
    }


def _extract_gdn_retrained(payload: Dict[str, Any]) -> Optional[Dict[str, Optional[float]]]:
    cls = payload.get("classification_metrics")
    sen = payload.get("sensor_attribution_metrics")
    if not isinstance(cls, dict) or not isinstance(sen, dict):
        return None

    return {
        "num_windows": None,
        "window_acc": _safe_float(cls.get("accuracy")),
        "window_precision": _safe_float(cls.get("precision")),
        "window_recall": _safe_float(cls.get("recall")),
        "window_f1": _safe_float(cls.get("f1_score")),
        "sensor_precision": _safe_float(sen.get("precision")),
        "sensor_recall": _safe_float(sen.get("recall")),
        "sensor_f1": _safe_float(sen.get("f1_score")),
        "fault_type_acc": None,
        "bertscore_f1": None,
    }


def _overall_score(row: Dict[str, Any]) -> Optional[float]:
    values = [row.get("window_f1"), row.get("sensor_f1"), row.get("fault_type_acc")]
    usable = [v for v in values if isinstance(v, float)]
    if not usable:
        return None
    return sum(usable) / len(usable)


def _fmt(value: Any, decimals: int = 4) -> str:
    if not isinstance(value, float):
        return "-"
    return f"{value:.{decimals}f}"


def _discover_inputs(pattern: str) -> List[Path]:
    paths = [Path(p) for p in sorted(glob.glob(pattern))]
    return [
        p for p in paths
        if p.is_file() and "examples_comparison" not in p.name
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise overall metrics for each method output JSON."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Result JSON files. If omitted, --glob is used.",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default="results-new/*.json",
        help="Glob used when no positional inputs are provided (default: results-new/*.json).",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs] if args.inputs else _discover_inputs(args.glob_pattern)
    if not input_paths:
        raise SystemExit("No result files found.")

    rows: List[Dict[str, Any]] = []
    for path in input_paths:
        try:
            with path.open("r") as f:
                payload = json.load(f)
        except Exception as exc:
            print(f"Skipping {path}: failed to read JSON ({exc})")
            continue

        parsed = (
            _extract_unified(payload)
            or _extract_legacy(payload)
            or _extract_gdn_retrained(payload)
        )
        if parsed is None:
            print(f"Skipping {path}: unsupported schema.")
            continue

        row: Dict[str, Any] = {
            "method": _method_name(payload, path),
            "file": str(path),
            **parsed,
        }
        row["overall_score"] = _overall_score(row)
        rows.append(row)

    if not rows:
        raise SystemExit("No supported result files to summarise.")

    rows.sort(key=lambda r: (r["overall_score"] is not None, r["overall_score"]), reverse=True)

    headers = [
        "method", "overall", "win_f1", "win_acc", "sen_f1", "sen_p", "sen_r", "fault_acc", "bert_f1", "n", "file"
    ]
    print("\t".join(headers))
    for row in rows:
        print("\t".join([
            row["method"],
            _fmt(row["overall_score"]),
            _fmt(row["window_f1"]),
            _fmt(row["window_acc"]),
            _fmt(row["sensor_f1"]),
            _fmt(row["sensor_precision"]),
            _fmt(row["sensor_recall"]),
            _fmt(row["fault_type_acc"]),
            _fmt(row["bertscore_f1"]),
            str(int(row["num_windows"])) if isinstance(row["num_windows"], float) else "-",
            row["file"],
        ]))


if __name__ == "__main__":
    main()
