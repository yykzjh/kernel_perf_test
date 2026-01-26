from __future__ import annotations

import json
import os
import random
import shutil
import platform
from datetime import timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kernel_perf_test import utils


def _setup_low_latency_env() -> None:
    # Single-node NVLink: enable low-latency optimizations.
    os.environ.setdefault("ACCL_LOW_LATENCY_OPTIMIZE", "1")
    # Keep dispatch/combine warp groups.
    os.environ.setdefault("ACCL_DISPATCH_NUM_WARP_GROUPS", "4")
    os.environ.setdefault("ACCL_COMBINE_NUM_WARP_GROUPS", "4")


def _create_test_data_low_latency(
    *,
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    num_topk: int,
    rank: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Test data for perf-only runs (no correctness checks):
    returns (x, topk_idx).
    """
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    x = torch.ones((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    # Mask a few positions to mimic sparse routing (avoid fully-dense topk).
    num_masks = min(10, num_tokens)
    if num_masks > 0:
        for _ in range(num_masks):
            topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1

    return x, topk_idx


def _init_deepep_low_latency_wrapper(
    *,
    world_size: int,
    local_rank: int,
    hidden_size: int,
    num_experts: int,
    max_batch_size_per_rank: int,
):
    """
    Initialize a DeepEP buffer in low-latency mode using the native deep_ep API
    (no rtp_llm wrappers).
    """
    torch.cuda.set_device(local_rank)
    torch.set_default_device(f"cuda:{local_rank}")

    # Distributed init (torchrun-friendly env://).
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=180),
        )

    dist.barrier()

    # Native deep_ep buffer init.
    try:
        from deep_ep import Buffer as DeepEPBuffer
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import deep_ep. Please ensure deep_ep is installed/built and visible in the current Python environment."
        ) from e

    # Low-latency sizing parameters.
    ll_num_max_token_per_rank = utils.calc_low_latency_max_token_per_rank(max_batch_size_per_rank, tp_size=1)
    num_rdma_bytes = DeepEPBuffer.get_low_latency_rdma_size_hint(
        ll_num_max_token_per_rank,
        hidden_size,
        world_size,
        num_experts,
    )

    # Match rtp_llm defaults: enable low-latency optimize + compute stream.
    os.environ.setdefault("ACCL_LOW_LATENCY_OPTIMIZE", "1")
    os.environ.setdefault("ACCL_LOW_LATENCY_USE_COMPUTE_STREAM", "1")

    def _allow_mnnvl() -> bool:
        # Equivalent to rtp_llm.allow_mnnvl(): aarch64 + SM100.
        try:
            is_sm_100 = torch.cuda.get_device_capability()[0] in [10]
        except Exception:
            is_sm_100 = False
        return ("aarch64" in platform.machine()) and is_sm_100

    init_kwargs = {
        "group": dist.group.WORLD,
        "num_nvl_bytes": 0,
        "num_rdma_bytes": int(num_rdma_bytes),
        "low_latency_mode": True,
        "num_qps_per_rank": int(num_experts / world_size),
        "allow_nvlink_for_low_latency_mode": True,
        "allow_mnnvl": bool(_allow_mnnvl()),
    }

    if local_rank == 0:
        print(
            f"[DeepEP] Allocating low-latency buffer: {int(num_rdma_bytes) / 1e6:.2f} MB | "
            f"ll_num_max_token_per_rank={ll_num_max_token_per_rank}, hidden_size={hidden_size}, "
            f"ep_size={world_size}, num_experts={num_experts}",
            flush=True,
        )

    buffer = DeepEPBuffer(**init_kwargs)  # type: ignore[misc]
    dist.barrier()
    return buffer, ll_num_max_token_per_rank


def _get_output_sub_dir_path(
    output_dir_path: str,
    *,
    ep_size: int,
    hidden_size: int,
    num_experts: int,
    num_topk: int,
    num_peo_rounds: int,
) -> str:
    return os.path.join(
        output_dir_path,
        (
            f"ep_size-{ep_size}_hidden_size-{hidden_size}_num_experts-{num_experts}_"
            f"num_topk-{num_topk}_num_peo_rounds-{num_peo_rounds}"
        ),
    )


@dataclass(frozen=True)
class DeepepDispatchBenchConfig:
    ep_size: int
    hidden_size: int
    num_experts: int
    num_topk: int
    num_peo_rounds: int
    batch_size_per_rank_list: List[int]
    num_warmups: int
    num_tests: int
    seed: int
    enable_trace: bool
    output_dir_path: str


def _run_worker(rank: int, cfg: DeepepDispatchBenchConfig, master_port: int) -> None:
    """
    One worker run: under a given ep_size (= world_size), sweep all batch_size_per_rank values.
    """
    _setup_low_latency_env()

    local_rank = rank
    torch.cuda.set_device(local_rank)
    torch.set_default_device(f"cuda:{local_rank}")

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{master_port}",
        rank=rank,
        world_size=cfg.ep_size,
        timeout=timedelta(seconds=180),
    )

    # Total SMs on the device (used for per-round SM budgeting).
    num_device_total_sms = utils.get_num_device_sms()
    sm_per_round = num_device_total_sms // cfg.num_peo_rounds
    if sm_per_round <= 0:
        raise RuntimeError(
            f"Invalid SM budget: total_sms={num_device_total_sms}, num_peo_rounds={cfg.num_peo_rounds} "
            f"-> total_sms // num_peo_rounds = {sm_per_round} (must be >= 1)"
        )

    buffer, ll_num_max_token_per_rank = _init_deepep_low_latency_wrapper(
        world_size=cfg.ep_size,
        local_rank=local_rank,
        hidden_size=cfg.hidden_size,
        num_experts=cfg.num_experts,
        max_batch_size_per_rank=max(cfg.batch_size_per_rank_list),
    )

    output_sub_dir_path = _get_output_sub_dir_path(
        cfg.output_dir_path,
        ep_size=cfg.ep_size,
        hidden_size=cfg.hidden_size,
        num_experts=cfg.num_experts,
        num_topk=cfg.num_topk,
        num_peo_rounds=cfg.num_peo_rounds,
    )
    output_sub_dir = Path(output_sub_dir_path)
    trace_dir = output_sub_dir / "traces"

    if rank == 0:
        if output_sub_dir.exists():
            shutil.rmtree(output_sub_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    batch_sizes: List[int] = list(cfg.batch_size_per_rank_list)
    normal_avg_us: List[float] = []
    peo_avg_us: List[float] = []
    normal_rank_stats: List[dict] = []
    peo_rank_stats: List[dict] = []

    for bs in batch_sizes:
        x, topk_idx = _create_test_data_low_latency(
            num_tokens=bs,
            hidden_size=cfg.hidden_size,
            num_experts=cfg.num_experts,
            num_topk=cfg.num_topk,
            rank=rank,
            seed=cfg.seed,
        )

        def normal_dispatch_once() -> None:
            # NORMAL low-latency dispatch (no recv hook). Wait on the completion event.
            _packed_recv_x, _packed_recv_count, _handle, event, _hook = buffer.low_latency_dispatch(
                x=x,
                topk_idx=topk_idx,
                num_max_dispatch_tokens_per_rank=ll_num_max_token_per_rank,
                num_experts=cfg.num_experts,
                use_fp8=True,
                round_scale=False,
                use_ue8m0=False,
                async_finish=True,
                return_recv_hook=False,
            )
            event.current_stream_wait()

        # Capture NORMAL into a CUDA Graph.
        dist.barrier()
        normal_graph = utils.capture_graph(normal_dispatch_once, num_warmups=cfg.num_warmups)
        dist.barrier()

        def peo_dispatch_once() -> None:
            """
            PEO low-latency dispatch (single round):
            - Only execute round_0.
            """
            _r0_packed_recv_x, _r0_packed_recv_count, _r0_handle, _r0_event, r0_hook = buffer.low_latency_dispatch(
                x=x,
                topk_idx=topk_idx,
                num_max_dispatch_tokens_per_rank=ll_num_max_token_per_rank,
                num_experts=cfg.num_experts,
                use_fp8=True,
                round_scale=False,
                use_ue8m0=False,
                async_finish=False,
                return_recv_hook=True,
                use_expert_overlap=True,
                round_id=0,
                send_num_sms=sm_per_round,
                recv_num_sms=sm_per_round,
                num_rounds=cfg.num_peo_rounds,
                hook_use_comm_stream=False,
            )
            r0_hook()

        # Capture PEO into a CUDA Graph (global capture mode can include multi-stream work).
        dist.barrier()
        peo_graph = utils.capture_graph(peo_dispatch_once, num_warmups=cfg.num_warmups)
        dist.barrier()

        # Measure with bench_kineto using cache.zero_() as the delimiter kernel.
        # Important: do NOT call dist.barrier() inside the measured fn, otherwise the barrier kernel may pollute traces.
        # We only synchronize at the boundaries between profiling runs.
        dist.barrier()
        n_trace_path = str(trace_dir / f"normal_rank{rank}_bs{bs}.json") if cfg.enable_trace else None
        n_avg_us_rank = float(
            utils.bench_kineto(
                lambda: normal_graph.replay(),
                num_warmups=cfg.num_warmups,
                num_tests=cfg.num_tests,
                suppress_kineto_output=True,
                trace_path=n_trace_path,
                position_shift=(3, 1),
            )
        )
        dist.barrier()

        dist.barrier()
        p_trace_path = str(trace_dir / f"peo_rank{rank}_bs{bs}.json") if cfg.enable_trace else None
        p_avg_us_rank = float(
            utils.bench_kineto(
                lambda: peo_graph.replay(),
                num_warmups=cfg.num_warmups,
                num_tests=cfg.num_tests,
                suppress_kineto_output=True,
                trace_path=p_trace_path,
                position_shift=(3, 1),
            )
        )
        dist.barrier()

        # Free graph objects per batch size to keep memory bounded.
        del normal_graph
        del peo_graph

        # Reduce to a world-average latency (mean of per-rank averages).
        n_avg = torch.tensor([n_avg_us_rank], device="cuda", dtype=torch.float64)
        p_avg = torch.tensor([p_avg_us_rank], device="cuda", dtype=torch.float64)
        dist.all_reduce(n_avg, op=dist.ReduceOp.SUM)
        dist.all_reduce(p_avg, op=dist.ReduceOp.SUM)
        n_avg_us_all = float((n_avg / cfg.ep_size).item())
        p_avg_us_all = float((p_avg / cfg.ep_size).item())

        normal_avg_us.append(n_avg_us_all)
        peo_avg_us.append(p_avg_us_all)

        normal_rank_stats.append(
            {
                "rank": rank,
                "batch_size_per_rank": bs,
                "avg_us": float(n_avg_us_rank),
            }
        )
        peo_rank_stats.append(
            {
                "rank": rank,
                "batch_size_per_rank": bs,
                "avg_us": float(p_avg_us_rank),
            }
        )

        dist.barrier()

    # Save results & plot (rank 0 only).
    if rank == 0:
        output_sub_dir.mkdir(parents=True, exist_ok=True)
        result_json_path = output_sub_dir / "dispatch_latency_results.json"
        result_json_path.write_text(
            json.dumps(
                {
                    "config": {
                        "ep_size": cfg.ep_size,
                        "hidden_size": cfg.hidden_size,
                        "num_experts": cfg.num_experts,
                        "num_topk": cfg.num_topk,
                        "num_peo_rounds": cfg.num_peo_rounds,
                        "ll_num_max_token_per_rank": ll_num_max_token_per_rank,
                        "num_warmups": cfg.num_warmups,
                        "num_tests": cfg.num_tests,
                        "seed": cfg.seed,
                    },
                    "batch_size_per_rank": batch_sizes,
                    "normal_avg_us": normal_avg_us,
                    "peo_avg_us": peo_avg_us,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        # CSV.
        csv_path = output_sub_dir / "dispatch_latency_results.csv"
        lines = ["batch_size_per_rank,normal_avg_us,peo_avg_us"]
        for b, n, p in zip(batch_sizes, normal_avg_us, peo_avg_us):
            lines.append(f"{b},{n:.6f},{p:.6f}")
        csv_path.write_text("\n".join(lines))

        # Plot (per ep_size).
        fig, ax = plt.subplots(figsize=(12.5, 7.0))
        ax.plot(batch_sizes, normal_avg_us, marker="o", linewidth=2, label="Normal low_latency")
        ax.plot(
            batch_sizes,
            peo_avg_us,
            marker="o",
            linewidth=2,
            label=f"PEO low_latency (rounds={cfg.num_peo_rounds})",
        )
        ax.set_xlabel("batch_size_per_rank")
        ax.set_ylabel("Latency / us")
        ax.set_title(f"DeepEP low_latency dispatch latency (ep_size={cfg.ep_size})")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best")
        fig.tight_layout()
        fig_path = output_sub_dir / "dispatch_latency_vs_batch_size.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)

        # Per-rank stats (debug).
        (output_sub_dir / "normal_rank_stats.json").write_text(
            json.dumps(normal_rank_stats, indent=2, ensure_ascii=False)
        )
        (output_sub_dir / "peo_rank_stats.json").write_text(json.dumps(peo_rank_stats, indent=2, ensure_ascii=False))

        print(f"[OK] Saved results to: {output_sub_dir}", flush=True)

    # Cleanup.
    dist.barrier()
    try:
        del buffer
    except Exception:
        pass
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def analyse_deepep_low_latency_dispatch_latency(
    *,
    ep_size_list: List[int],
    batch_size_per_rank_list: List[int],
    hidden_size: int = 6144,
    num_experts: int = 160,
    num_topk: int = 8,
    num_peo_rounds_list: List[int] = [2, 4, 5, 8, 10],
    num_warmups: int = 2,
    num_tests: int = 5,
    seed: int = 1,
    enable_trace: bool = False,
    output_dir_path: str = "./deepep_peo_bs_latency_analysis_result",
    master_port: int = 29500,
) -> Path:
    """
    Batch benchmark over the parameter grid:
    - ep_size_list × num_peo_rounds_list × batch_size_per_rank_list

    Outputs:
    - per-(ep_size, num_peo_rounds) result folders (JSON/CSV/trace/plot)
    - aggregated per-ep_size plots comparing different num_peo_rounds
    - a flat matrix CSV/JSON for downstream analysis

    Notes:
    - This is a function-style entrypoint (no CLI parsing), mirroring deep_gemm analysis scripts.
    - Distributed runs are launched via torch.multiprocessing.spawn on a single node.
    """
    if not ep_size_list:
        raise ValueError("ep_size_list must be non-empty")
    if not batch_size_per_rank_list:
        raise ValueError("batch_size_per_rank_list must be non-empty")

    out_root = Path(output_dir_path)
    out_root.mkdir(parents=True, exist_ok=True)

    ep_sizes = list(ep_size_list)
    rounds_list = list(num_peo_rounds_list)
    num_visible_gpus = torch.cuda.device_count()
    if num_visible_gpus <= 0:
        raise RuntimeError("No visible CUDA devices. Please check your CUDA environment.")

    regression_figures: List[str] = []

    for ep_size in ep_sizes:
        if ep_size <= 0:
            raise ValueError(f"Invalid ep_size: {ep_size} (must be >= 1)")
        if ep_size > num_visible_gpus:
            raise RuntimeError(
                f"ep_size={ep_size} exceeds visible CUDA devices ({num_visible_gpus}). "
                f"Adjust CUDA_VISIBLE_DEVICES or run on a node with more GPUs."
            )

        if num_experts % ep_size != 0:
            raise RuntimeError(f"num_experts={num_experts} must be divisible by ep_size={ep_size}")
        num_local_experts = num_experts // ep_size

        run_idx = 0
        for num_peo_rounds in rounds_list:
            # Constraint: num_local_experts must be divisible by num_peo_rounds.
            if num_local_experts % num_peo_rounds != 0:
                continue

            # NOTE:
            # We intentionally do NOT override CUDA_VISIBLE_DEVICES here.
            # The run uses local ranks [0..ep_size-1], so make sure at least `ep_size` GPUs are visible.

            cfg = DeepepDispatchBenchConfig(
                ep_size=ep_size,
                hidden_size=hidden_size,
                num_experts=num_experts,
                num_topk=num_topk,
                num_peo_rounds=num_peo_rounds,
                batch_size_per_rank_list=list(batch_size_per_rank_list),
                num_warmups=num_warmups,
                num_tests=num_tests,
                seed=seed,
                enable_trace=enable_trace,
                output_dir_path=output_dir_path,
            )
            # Use a deterministic per-run port derived from master_port.
            # This keeps the init IP fixed (127.0.0.1) and avoids reuse conflicts across runs.
            port = int(master_port) + run_idx
            if port <= 0 or port > 65535:
                raise ValueError(f"Invalid master port: {port} (base={master_port}, run_idx={run_idx})")
            mp.spawn(
                _run_worker,
                args=(cfg, port),
                nprocs=ep_size,
                join=True,
            )
            run_idx += 1

            # Compare PEO vs Normal; if PEO is slower for any batch size, record the per-config figure path.
            output_sub_dir_path = _get_output_sub_dir_path(
                output_dir_path,
                ep_size=ep_size,
                hidden_size=hidden_size,
                num_experts=num_experts,
                num_topk=num_topk,
                num_peo_rounds=num_peo_rounds,
            )
            result_path = Path(output_sub_dir_path) / "dispatch_latency_results.json"
            if not result_path.exists():
                raise FileNotFoundError(f"Missing result file: {result_path}")
            payload = json.loads(result_path.read_text())
            normal_avg_us = payload.get("normal_avg_us", [])
            peo_avg_us = payload.get("peo_avg_us", [])
            if not isinstance(normal_avg_us, list) or not isinstance(peo_avg_us, list):
                raise RuntimeError(f"Invalid result format in: {result_path}")
            any_slower = any(float(p) > float(n) for p, n in zip(peo_avg_us, normal_avg_us))
            if any_slower:
                fig_path = str(Path(output_sub_dir_path) / "dispatch_latency_vs_batch_size.png")
                regression_figures.append(f"ep_size={ep_size}, num_peo_rounds={num_peo_rounds} -> {fig_path}")

    if regression_figures:
        print("\n".join(regression_figures), flush=True)

    return out_root


if __name__ == "__main__":
    analyse_deepep_low_latency_dispatch_latency(
        ep_size_list=[2, 4, 8],
        batch_size_per_rank_list=[
            1,
            4,
            8,
            16,
            24,
            32,
            48,
            64,
            80,
            96,
            112,
            128,
            160,
            192,
            224,
            256,
        ],
        hidden_size=6144,
        num_experts=160,
        num_topk=8,
        num_peo_rounds_list=[2, 4],
        num_warmups=2,
        num_tests=5,
        seed=1,
        enable_trace=True,
        output_dir_path="./deepep_peo_bs_latency_analysis_result",
        master_port=29500,
    )
