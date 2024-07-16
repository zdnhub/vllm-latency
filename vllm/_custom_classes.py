import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    # ruff: noqa: F401
    import vllm._core_C
except ImportError as e:
    logger.warning("Failed to import from vllm._core_C with %r", e)

ScalarType = torch.classes._core_C.ScalarType
