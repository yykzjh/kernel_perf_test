import torch
from typing import Tuple


def get_num_device_sms() -> int:
    assert torch.cuda.is_available()
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return props.multi_processor_count


def get_compute_capability() -> Tuple[int, int]:
    assert torch.cuda.is_available()
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return props.major, props.minor
