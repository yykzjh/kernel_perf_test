import os
import sys
import json
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

import torch


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def bench(fn, num_warmups: int = 50, num_tests: int = 50, post_fn=None):
    # Flush L2 cache with 256 MB data
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Flush L2
    cache.zero_()

    lhs = torch.randn((8192, 8192), dtype=torch.float, device="cuda")
    rhs = torch.randn((8192, 8192), dtype=torch.float, device="cuda")
    lhs @ rhs

    # Testing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        # Record
        start_events[i].record()
        fn()
        end_events[i].record()
        if post_fn is not None:
            post_fn()
    torch.cuda.synchronize()

    times = np.array([s.elapsed_time(e) for s, e in zip(start_events, end_events)])[1:]
    return np.average(times) * 1e3, np.min(times) * 1e3, np.max(times) * 1e3


def bench_kineto(
    fn,
    num_warmups: int = 50,
    num_tests: int = 30,
    suppress_kineto_output: bool = False,
    trace_path: Optional[str] = None,
    num_kernels_per_period: Optional[List[int]] = None,
    kernel_ranges: Optional[List[Tuple[str, str]]] = None,
):
    # Flush L2 cache with 256 MB data
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    # Warmup
    for _ in range(num_warmups):
        fn()
    # Add a large kernel to eliminate the CPU launch overhead
    lhs = torch.randn((8192, 8192), dtype=torch.float, device="cuda")
    rhs = torch.randn((8192, 8192), dtype=torch.float, device="cuda")

    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule) as prof:
            lhs @ rhs
            for i in range(num_tests):
                # Record
                cache.zero_()
                fn()
            torch.cuda.synchronize()
            prof.step()
    # Save chrome traces
    if trace_path is not None:
        prof.export_chrome_trace(trace_path)

    if kernel_ranges is None:
        return

    # Create a temporary trace file
    trace_path_to_use = trace_path
    if trace_path is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        trace_path_to_use = temp_file.name
        temp_file.close()
        prof.export_chrome_trace(trace_path_to_use)
    # Check parameters
    if num_kernels_per_period is None:
        num_kernels_per_period = [1] * len(kernel_ranges)
    else:
        assert len(num_kernels_per_period) == len(
            kernel_ranges
        ), "num_kernels_per_period must be the same length as kernel_ranges"
    # Parse the trace events
    kernel_range_times = parse_trace_events(
        trace_path=trace_path_to_use,
        kernel_ranges=kernel_ranges,
        num_kernels_per_period=num_kernels_per_period,
    )
    # Clean up the temporary trace file
    if trace_path is None:
        os.remove(trace_path_to_use)

    # Return execution durations
    return kernel_range_times


def parse_trace_events(
    trace_path: str,
    kernel_ranges: List[Tuple[str, str]],
    num_kernels_per_period: List[int],
) -> List[List[float]]:
    # Load the trace events
    profile_data = json.loads(Path(trace_path).read_text())
    trace_events = profile_data["traceEvents"]
    # Filter the trace events to only include the kernel events
    trace_events = [event for event in trace_events if (event["ph"] == "X" and event["cat"] == "kernel")]
    # Sort the trace events by timestamp
    trace_events = sorted(trace_events, key=lambda event: event["ts"])
    # Strip the trace events to only include the full kernel events
    matching_indices = [
        i
        for i, event in enumerate(trace_events)
        if (
            "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, at::native::FillFunctor<int>, std::array<char*, 1ul>)"
            in event["name"]
        )
    ]
    assert len(matching_indices) > 1, "There should be at least two matching indices"
    del trace_events[matching_indices[-1] + 1 :]
    del trace_events[: matching_indices[0]]
    # Iterative total kernel_range
    kernel_range_times = []
    for i, kernel_range in enumerate(kernel_ranges):
        # Calculate the duration of the kernel range
        start_kernel_events = [event for event in trace_events if kernel_range[0] in event["name"]]
        end_kernel_events = [event for event in trace_events if kernel_range[1] in event["name"]]
        if len(end_kernel_events) == 0:
            end_kernel_events = [event for event in trace_events if kernel_range[0] in event["name"]]
        if len(start_kernel_events) > len(end_kernel_events) and len(start_kernel_events) % len(end_kernel_events) == 0:
            start_kernel_events = start_kernel_events[:: len(start_kernel_events) // len(end_kernel_events)]
        if len(end_kernel_events) > len(start_kernel_events) and len(end_kernel_events) % len(start_kernel_events) == 0:
            end_kernel_events = end_kernel_events[:: len(end_kernel_events) // len(start_kernel_events)]
        assert len(start_kernel_events) == len(
            end_kernel_events
        ), f"The number of start event: {len(start_kernel_events)} and the number of end event: {len(end_kernel_events)} must be the same"
        kernel_range_durations = [
            end_kernel_events[i]["ts"] + end_kernel_events[i]["dur"] - start_kernel_events[i]["ts"]
            for i in range(len(start_kernel_events))
        ]
        # Process the kernel range durations
        num_kernel_patterns = len(kernel_range_durations) // num_kernels_per_period[i]
        kernel_range_times.append(
            [
                sum(kernel_range_durations[j :: num_kernels_per_period[i]]) / num_kernel_patterns
                for j in range(num_kernels_per_period[i])
            ]
        )
    return kernel_range_times


if __name__ == "__main__":
    kernel_range_times = parse_trace_events(
        trace_path=r"./torch_cuda_profiler/replicated_linear_moe_gate/deepgemm_masked_moe_ffn_trace_num_local_experts-24_expected_m-86.json",
        kernel_ranges=[
            (
                "void (anonymous namespace)::elementwise_kernel_with_index",
                "void deep_gemm::sm100_fp8_gemm_1d1d_impl",
            )
        ],
        num_kernels_per_period=[1],
    )
    print(kernel_range_times)

    kernel_range_times = parse_trace_events(
        trace_path=r"./torch_cuda_profiler/replicated_linear_moe_gate/replicated_linear_moe_gate_trace_batch_size-256.json",
        kernel_ranges=[
            (
                "nvjet",
                "void cublasLt::splitKreduce_kernel",
            )
        ],
        num_kernels_per_period=[1],
    )
    print(kernel_range_times)
