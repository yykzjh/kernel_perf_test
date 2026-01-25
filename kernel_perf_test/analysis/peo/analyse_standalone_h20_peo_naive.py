from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# Config parsing (task dir)
# =========================


@dataclass(frozen=True)
class TaskConfig:
    task_dir_name: str
    batch_size: int
    overlap: Optional[int]  # enable_peo_level, None for NORMAL
    num_peo_rounds: Optional[int]
    deep_ep_num_sm: Optional[int]


_BATCH_RE = re.compile(r"batch_size_list-\[(\d+)\]")
_OVERLAP_RE = re.compile(r"enable_peo_level-(\d+)")
_ROUNDS_RE = re.compile(r"num_peo_rounds-(\d+)")
_SM_RE = re.compile(r"deep_ep_num_sm-(\d+)")


def parse_task_dir_name(task_dir_name: str) -> TaskConfig:
    batch_m = _BATCH_RE.search(task_dir_name)
    if not batch_m:
        raise ValueError(f"Cannot parse batch_size from task dir name: {task_dir_name}")
    batch_size = int(batch_m.group(1))

    overlap_m = _OVERLAP_RE.search(task_dir_name)
    overlap = int(overlap_m.group(1)) if overlap_m else None

    rounds_m = _ROUNDS_RE.search(task_dir_name)
    num_peo_rounds = int(rounds_m.group(1)) if rounds_m else None

    sm_m = _SM_RE.search(task_dir_name)
    deep_ep_num_sm = int(sm_m.group(1)) if sm_m else None

    return TaskConfig(
        task_dir_name=task_dir_name,
        batch_size=batch_size,
        overlap=overlap,
        num_peo_rounds=num_peo_rounds,
        deep_ep_num_sm=deep_ep_num_sm,
    )


# =========================
# MoE latency extraction
# =========================


_KERNEL_FAKE_BALANCE_NAME = (
    "void rtp_llm::fakeBalanceExpertKernel<long>(long*, float*, int, int, int, int, int)"
)
_KERNEL_DISPATCH_NAME = (
    "void deep_ep::internode_ll::dispatch_opt1_hidden6144<1, false, 1, false, false, 6144, false, false>(void*, void*, float*, int*, long*, int*, int*, int*, int*, int*, int*, int*, int*, long*, void*, int*, void*, void**, unsigned long, unsigned long, void const*, float const*, long*, float const*, int*, int*, int*, int*, int*, int*, int, int*, int, int, int, int, int, bool, int, int, int, int, int, int, int, int, bool, int, bool, int, int, void*, int*, int*, int*, int*, int*, int, bool, int, void*, void*, int, void*, int, void*, int, void*, int, int, nccl_call*, bool, int, int, int, int)"
)
_KERNEL_COMBINE_NAME = (
    "void deep_ep::internode_ll::combine_opt1_hidden6144<false, false, 1, false, false, 6144, 11, 4, false>(void*, void*, int*, void*, void**, unsigned long, unsigned long, void const*, long const*, float const*, float const*, int const*, int const*, int const*, int const*, int const*, int const*, int const*, long const*, int*, long*, int*, int, int*, int, int*, int*, int*, int*, int*, int, int, int, int, int, int, bool, int, int, int, int, int, int, int, int, bool, int, bool, bool, int, int, int, int, bool, nccl_call*)"
)
_KERNEL_RMSNORM_NAME = (
    "void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16>(__nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, unsigned int, float, float)"
)


@dataclass(frozen=True)
class KernelEvent:
    name: str
    ts_us: float
    dur_us: float

    @property
    def end_ts_us(self) -> float:
        return self.ts_us + self.dur_us


def _iter_kernel_events(trace_events: list[dict[str, Any]]) -> list[KernelEvent]:
    out: list[KernelEvent] = []
    for ev in trace_events:
        if ev.get("ph") != "X":
            continue
        if ev.get("cat") != "kernel":
            continue
        name = ev.get("name")
        ts = ev.get("ts")
        dur = ev.get("dur")
        if not isinstance(name, str):
            continue
        if not isinstance(ts, (int, float)) or not isinstance(dur, (int, float)):
            continue
        out.append(KernelEvent(name=name, ts_us=float(ts), dur_us=float(dur)))
    # The trace is typically already time-ordered, but sorting makes us robust.
    out.sort(key=lambda e: e.ts_us)
    return out


def _extract_moe_durations_us_from_kernels(kernels: list[KernelEvent]) -> list[float]:
    """
    Extract MoE latency from kernel timeline (sorted by ts), using *adjacent kernel*
    boundaries to avoid mismatching:

    - Start boundary: two consecutive kernels are
      1) fakeBalance
      2) dispatch_opt1_hidden6144
      Use dispatch.ts as MoE start time.

    - End boundary: two consecutive kernels are
      1) combine_opt1_hidden6144
      2) FusedAddRMSNormKernel
      Use combine.(ts + dur) as MoE end time.

    MoE latency = end_time - start_time (unit: us)
    """
    durations: list[float] = []
    i = 0
    n = len(kernels)
    while i + 1 < n:
        k0 = kernels[i]
        k1 = kernels[i + 1]

        # Find start pair: fakeBalance -> dispatch
        if not (k0.name == _KERNEL_FAKE_BALANCE_NAME and k1.name == _KERNEL_DISPATCH_NAME):
            i += 1
            continue

        moe_start_ts = k1.ts_us

        # Find the next end pair: combine -> rmsnorm, after the dispatch
        j = i + 1
        found_end = False
        while j + 1 < n:
            c0 = kernels[j]
            c1 = kernels[j + 1]
            if c0.name == _KERNEL_COMBINE_NAME and c1.name == _KERNEL_RMSNORM_NAME:
                moe_end_ts = c0.end_ts_us
                dur = moe_end_ts - moe_start_ts
                if dur > 0:
                    durations.append(float(dur))
                i = j + 1  # continue after rmsnorm
                found_end = True
                break
            j += 1

        if not found_end:
            break

    return durations


def extract_moe_durations_us_from_trace_json(trace_json_path: Path) -> list[float]:
    with trace_json_path.open("r") as f:
        data = json.load(f)
    trace_events = data.get("traceEvents", [])
    if not isinstance(trace_events, list):
        return []
    kernels = _iter_kernel_events(trace_events)
    return _extract_moe_durations_us_from_kernels(kernels)


# =========================
# Robust aggregation
# =========================


def _filter_outliers(values: np.ndarray) -> np.ndarray:
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if values.size < 6:
        return values

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        # Fall back to percentile trimming
        lo, hi = np.percentile(values, [5, 95])
        return values[(values >= lo) & (values <= hi)]

    lo_iqr = q1 - 1.5 * iqr
    hi_iqr = q3 + 1.5 * iqr
    lo_p, hi_p = np.percentile(values, [5, 95])
    lo = max(lo_iqr, lo_p)
    hi = min(hi_iqr, hi_p)
    kept = values[(values >= lo) & (values <= hi)]
    return kept if kept.size >= 3 else values


def compute_task_moe_latency_us(task_dir_path: Path) -> float:
    """
    One task: aggregate 8 wr traces. Each trace usually has 4 MoE durations;
    drop the first (warm-up / abnormal long) per-file, then merge all, drop outliers, mean.
    """
    trace_dir = task_dir_path / "trace_files"
    if not trace_dir.exists():
        raise FileNotFoundError(f"Missing trace_files in {task_dir_path}")

    json_paths = sorted(trace_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No trace json under {trace_dir}")

    all_durations: list[float] = []
    for p in json_paths:
        durs = extract_moe_durations_us_from_trace_json(p)
        if len(durs) >= 2:
            durs = durs[1:]  # drop first layer
        all_durations.extend(durs)

    if not all_durations:
        raise RuntimeError(f"No MoE durations extracted from {task_dir_path}")

    arr = np.asarray(all_durations, dtype=np.float64)
    arr = _filter_outliers(arr)
    return float(np.mean(arr))


# =========================
# Benchmark scanning
# =========================


@dataclass(frozen=True)
class BestResult:
    latency_us: float
    task_dir: str
    overlap: Optional[int]
    batch_size: int
    num_peo_rounds: Optional[int]
    deep_ep_num_sm: Optional[int]


def _iter_task_dirs(benchmark_dir: Path) -> list[Path]:
    return sorted(
        [p for p in benchmark_dir.iterdir() if p.is_dir() and p.name.startswith("Task_")],
        key=lambda p: p.name,
    )


def discover_batch_sizes_from_benchmark_dir(benchmark_dir: Path) -> list[int]:
    batch_sizes: set[int] = set()
    for task_dir in _iter_task_dirs(benchmark_dir):
        try:
            cfg = parse_task_dir_name(task_dir.name)
        except Exception:
            continue
        batch_sizes.add(cfg.batch_size)
    return sorted(batch_sizes)


def find_benchmark_dir(experiment_root: Path, kind: str) -> Path:
    """
    kind: 'NORMAL' or 'PEO'
    """
    candidates = sorted(experiment_root.glob(f"Benchmark_*_{kind}_*"))
    if not candidates:
        raise FileNotFoundError(f"Cannot find Benchmark_*_{kind}_* under {experiment_root}")
    if len(candidates) > 1:
        # pick the latest by name (timestamp suffix)
        candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def collect_normal_results(normal_benchmark_dir: Path, batch_sizes: list[int]) -> dict[int, BestResult]:
    best: dict[int, BestResult] = {}
    for task_dir in _iter_task_dirs(normal_benchmark_dir):
        cfg = parse_task_dir_name(task_dir.name)
        if cfg.batch_size not in set(batch_sizes):
            continue
        latency = compute_task_moe_latency_us(task_dir)
        cur = best.get(cfg.batch_size)
        if (cur is None) or (latency < cur.latency_us):
            best[cfg.batch_size] = BestResult(
                latency_us=latency,
                task_dir=task_dir.name,
                overlap=None,
                batch_size=cfg.batch_size,
                num_peo_rounds=None,
                deep_ep_num_sm=None,
            )
    return best


def collect_peo_best_results(
    peo_benchmark_dir: Path, batch_sizes: list[int], overlaps: list[int]
) -> dict[int, dict[int, BestResult]]:
    best: dict[int, dict[int, BestResult]] = {o: {} for o in overlaps}
    batch_sizes_set = set(batch_sizes)
    overlaps_set = set(overlaps)

    for task_dir in _iter_task_dirs(peo_benchmark_dir):
        cfg = parse_task_dir_name(task_dir.name)
        if cfg.batch_size not in batch_sizes_set:
            continue
        if cfg.overlap is None or cfg.overlap not in overlaps_set:
            continue
        latency = compute_task_moe_latency_us(task_dir)
        cur = best[cfg.overlap].get(cfg.batch_size)
        if (cur is None) or (latency < cur.latency_us):
            best[cfg.overlap][cfg.batch_size] = BestResult(
                latency_us=latency,
                task_dir=task_dir.name,
                overlap=cfg.overlap,
                batch_size=cfg.batch_size,
                num_peo_rounds=cfg.num_peo_rounds,
                deep_ep_num_sm=cfg.deep_ep_num_sm,
            )
    return best


# =========================
# Plotting
# =========================


def plot_moe_latency_comparison(
    output_dir: Path,
    batch_sizes: list[int],
    normal_best: dict[int, BestResult],
    peo_best: dict[int, dict[int, BestResult]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "moe_latency_vs_batch_size.png"

    x = batch_sizes

    def y_for(best_map: dict[int, BestResult]) -> list[float]:
        y: list[float] = []
        for b in x:
            if b in best_map:
                y.append(best_map[b].latency_us)
            else:
                y.append(float("nan"))
        return y

    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    ax.plot(x, y_for(normal_best), marker="o", linewidth=2, label="NORMAL")

    overlap_to_color: dict[int, str] = {}
    overlap_to_y: dict[int, list[float]] = {}

    for overlap in sorted(peo_best.keys()):
        y = y_for(peo_best[overlap])
        (line,) = ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label=f"OVERLAP_{overlap}",
        )
        overlap_to_color[overlap] = line.get_color()
        overlap_to_y[overlap] = y

    # Annotate PEO points with best (num_peo_rounds, deep_ep_num_sm) as "(rounds, sm)".
    # To reduce overlaps:
    # - For each batch_size (same x), collect all overlap points, sort by y.
    # - Then distribute labels with alternating left/right and different y offsets.
    x_offsets = [-36, 36, -36, 36]  # points
    y_offsets = [-26, -8, 12, 30]  # points
    for idx_x, xi in enumerate(x):
        entries: list[tuple[float, int]] = []
        for overlap, ys in overlap_to_y.items():
            yi = ys[idx_x]
            if np.isfinite(yi) and (xi in peo_best.get(overlap, {})):
                entries.append((yi, overlap))
        if not entries:
            continue

        entries.sort(key=lambda t: t[0])  # low y first
        for rank, (yi, overlap) in enumerate(entries):
            br = peo_best[overlap].get(xi)
            if br is None or br.num_peo_rounds is None or br.deep_ep_num_sm is None:
                continue
            color = overlap_to_color.get(overlap, "black")

            xo = x_offsets[rank % len(x_offsets)]
            yo = y_offsets[rank % len(y_offsets)]
            # Add tiny jitter per x to avoid same-rank collisions across columns.
            xo = xo + (idx_x % 2) * 6

            ax.annotate(
                f"({br.num_peo_rounds}, {br.deep_ep_num_sm})",
                xy=(xi, yi),
                xytext=(xo, yo),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    fc="white",
                    ec=color,
                    alpha=0.85,
                    linewidth=0.9,
                ),
            )

    # Add a note explaining the meaning of "(rounds, sm)" annotations.
    ax.text(
        0.99,
        0.02,
        "Annotation (r, sm) means (num_peo_rounds, deep_ep_num_sm) for the best PEO config at that point.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.75, linewidth=0.8),
    )

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("MoE Latency / us")
    ax.set_title("MoE Latency vs Batch Size (NORMAL vs PEO overlap modes)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    return fig_path


def save_best_results_json(
    output_dir: Path,
    batch_sizes: list[int],
    normal_best: dict[int, BestResult],
    peo_best: dict[int, dict[int, BestResult]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "best_moe_latency_results.json"

    payload: dict[str, Any] = {
        "batch_sizes": batch_sizes,
        "normal": {str(k): asdict(v) for k, v in normal_best.items()},
        "peo": {str(o): {str(k): asdict(v) for k, v in d.items()} for o, d in peo_best.items()},
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return out_path


# =========================
# CLI
# =========================


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MoE latency: NORMAL vs PEO overlaps.")
    parser.add_argument(
        "--experiment-root",
        type=str,
        default="/home/yiyin.zjh/kernel_perf_test/Experiment_H20_Node1_PEO_20260122-160555-020254",
        help="Experiment root directory containing Benchmark_* subdirs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory for figures/results (default: <experiment-root>/peo_moe_latency_comparison).",
    )
    args = parser.parse_args()

    experiment_root = Path(args.experiment_root)
    output_dir = Path(args.output_dir) if args.output_dir else (experiment_root / "peo_moe_latency_comparison")

    normal_dir = find_benchmark_dir(experiment_root, "NORMAL")
    peo_dir = find_benchmark_dir(experiment_root, "PEO")

    print(f"[INFO] NORMAL benchmark: {normal_dir}")
    print(f"[INFO] PEO benchmark:    {peo_dir}")

    batch_sizes = sorted(
        set(discover_batch_sizes_from_benchmark_dir(normal_dir))
        | set(discover_batch_sizes_from_benchmark_dir(peo_dir))
    )
    if not batch_sizes:
        raise RuntimeError("No batch_size found from benchmark task directory names.")
    print(f"[INFO] Discovered batch sizes: {batch_sizes}")

    normal_best = collect_normal_results(normal_dir, batch_sizes=batch_sizes)
    peo_best = collect_peo_best_results(peo_dir, batch_sizes=batch_sizes, overlaps=[1, 2, 3, 4])

    # Print summary
    print("\n[RESULT] Best MoE latency (us)")
    for b in batch_sizes:
        n = normal_best.get(b)
        n_s = f"{n.latency_us:.3f} (task={n.task_dir})" if n else "N/A"
        line = [f"batch={b}", f"NORMAL={n_s}"]
        for o in [1, 2, 3, 4]:
            r = peo_best.get(o, {}).get(b)
            if r:
                line.append(
                    f"OVERLAP_{o}={r.latency_us:.3f} (rounds={r.num_peo_rounds}, sm={r.deep_ep_num_sm})"
                )
            else:
                line.append(f"OVERLAP_{o}=N/A")
        print(" | ".join(line))

    fig_path = plot_moe_latency_comparison(output_dir, batch_sizes, normal_best, peo_best)
    json_path = save_best_results_json(output_dir, batch_sizes, normal_best, peo_best)
    print(f"\n[INFO] Saved figure: {fig_path}")
    print(f"[INFO] Saved results: {json_path}")


if __name__ == "__main__":
    main()
