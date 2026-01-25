import functools
from typing import Generator
from contextlib import contextmanager

from kernel_perf_test.utils.module_util import has_module


@functools.cache
def has_deep_gemm() -> bool:
    """Whether the optional `deep_gemm` package is available."""
    return has_module("deep_gemm")


@contextmanager
def configure_deep_gemm_num_sms(num_sms: int) -> Generator[None, None, None]:
    """Configure the number of sms for deep gemm."""
    if not has_deep_gemm():
        raise RuntimeError(
            "DeepGEMM is not available. Please install the `deep_gemm` package to enable DeepGEMM kernels."
        )
    import deep_gemm

    # get original num sms
    original_num_sms = deep_gemm.get_num_sms()
    # set num sms
    deep_gemm.set_num_sms(num_sms)
    try:
        yield
    finally:
        # restore original num sms
        deep_gemm.set_num_sms(original_num_sms)
