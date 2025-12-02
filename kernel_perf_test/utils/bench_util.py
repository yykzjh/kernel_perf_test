import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Union, Optional

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
    kernel_names: Optional[Union[str, tuple]] = None,
    num_warmups: int = 50,
    num_tests: int = 30,
    suppress_kineto_output: bool = False,
    trace_path: Optional[str] = None,
    num_kernels_per_period: int = 1,
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

    # Initialize events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]

    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule) as prof:
            lhs @ rhs
            for i in range(num_tests):
                # Record
                cache.zero_()
                start_events[i].record()
                fn()
                end_events[i].record()
            torch.cuda.synchronize()
            prof.step()

    times = np.array([s.elapsed_time(e) for s, e in zip(start_events, end_events)])[1:]

    # Parse the profiling table
    if kernel_names is not None:
        assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
        is_tuple = isinstance(kernel_names, tuple)
        prof_lines = prof.key_averages().table(sort_by="cuda_time_total", max_name_column_width=100).split("\n")
        kernel_names = (kernel_names,) if isinstance(kernel_names, str) else kernel_names
        assert all([isinstance(name, str) for name in kernel_names])
        for name in kernel_names:
            assert (
                sum([name in line for line in prof_lines]) == 1
            ), f"Errors of the kernel {name} in the profiling table"

    # Save chrome traces
    if trace_path is not None:
        prof.export_chrome_trace(trace_path)

    if kernel_names is None:
        return np.average(times) * 1e3, np.min(times) * 1e3, np.max(times) * 1e3

    # Return average kernel durations
    units = {"ms": 1e3, "us": 1e6}
    kernel_durations = []
    for name in kernel_names:
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                for unit, scale in units.items():
                    if unit in time_str:
                        kernel_durations.append(float(time_str.replace(unit, "")) / scale)
                        break
                break

    # Expand the kernels by periods
    if num_kernels_per_period > 1:
        profile_data = json.loads(Path(trace_path).read_text())
        for i, kernel_name in enumerate(kernel_names):
            events = [event for event in profile_data["traceEvents"] if f"::{kernel_name}" in event["name"]]
            events = sorted(events, key=lambda event: event["ts"])
            durations = [event["dur"] / 1e6 for event in events]
            assert len(durations) % num_kernels_per_period == 0
            num_kernel_patterns = len(durations) // num_kernels_per_period
            kernel_durations[i] = [
                sum(durations[j::num_kernels_per_period]) / num_kernel_patterns for j in range(num_kernels_per_period)
            ]

    # Return execution durations
    return (
        (kernel_durations if is_tuple else kernel_durations[0]),
        np.average(times) * 1e3,
        np.min(times) * 1e3,
        np.max(times) * 1e3,
    )
