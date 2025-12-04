import os
import sys
import json
import tempfile
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

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
    position_shift: Tuple[int, int] = (1, 1),
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

    # Create a temporary trace file
    trace_path_to_use = trace_path
    if trace_path is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        trace_path_to_use = temp_file.name
        temp_file.close()
        prof.export_chrome_trace(trace_path_to_use)
    # Parse the trace events
    kernel_range_time = parse_trace_events(trace_path=trace_path_to_use, position_shift=position_shift)
    # Clean up the temporary trace file
    if trace_path is None:
        os.remove(trace_path_to_use)

    # Return execution durations
    return kernel_range_time


def parse_trace_events(trace_path: str, position_shift: Tuple[int, int] = (1, 1)) -> float:
    seperated_kernel_name = "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, at::native::FillFunctor<int>, std::array<char*, 1ul>)"
    # Load the trace events
    profile_data = json.loads(Path(trace_path).read_text())
    trace_events = profile_data["traceEvents"]
    # Filter the trace events to only include the kernel events
    trace_events = [event for event in trace_events if (event["ph"] == "X" and event["cat"] == "kernel")]
    # Sort the trace events by timestamp
    trace_events = sorted(trace_events, key=lambda event: event["ts"])
    # Strip the trace events to only include the full kernel events
    seperated_kernel_indices = [i for i, event in enumerate(trace_events) if (seperated_kernel_name in event["name"])]
    assert len(seperated_kernel_indices) > 1, "There should be at least two matching indices"
    del trace_events[seperated_kernel_indices[-1] + 1 :]
    del trace_events[: seperated_kernel_indices[0]]
    # Find seperated kernel indices
    seperated_kernel_indices = [i for i, event in enumerate(trace_events) if (seperated_kernel_name in event["name"])]
    # Calculate the duration of the kernel range
    start_kernel_event_indices = [
        seperated_kernel_indices[i] + position_shift[0] for i in range(0, len(seperated_kernel_indices) - 1)
    ]
    end_kernel_event_indices = [
        seperated_kernel_indices[i] - position_shift[1] for i in range(1, len(seperated_kernel_indices))
    ]
    kernel_range_durations = [
        trace_events[end_kernel_event_indices[i]]["ts"]
        + trace_events[end_kernel_event_indices[i]]["dur"]
        - trace_events[start_kernel_event_indices[i]]["ts"]
        for i in range(len(start_kernel_event_indices))
    ]
    return sum(kernel_range_durations) / len(kernel_range_durations)


if __name__ == "__main__":
    kernel_range_time = parse_trace_events(
        trace_path=r"./test_data/deepgemm_masked_moe_ffn_trace_num_local_experts-24_expected_m-86.json",
    )
    print(kernel_range_time)

    kernel_range_time = parse_trace_events(
        trace_path=r"./test_data/replicated_linear_moe_gate_trace_batch_size-256.json",
    )
    print(kernel_range_time)
