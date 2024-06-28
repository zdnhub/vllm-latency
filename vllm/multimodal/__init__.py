from .base import MultiModalData, MultiModalPlugin, EXTERNAL_MM_DATA_TYPE
from .registry import MultiModalRegistry

MULTIMODAL_REGISTRY = MultiModalRegistry()
"""
The global :class:`~MultiModalRegistry` is used by model runners to
dispatch data processing according to its modality and the target model.

See also:
    :ref:`input_processing_pipeline`
"""

__all__ = [
    "MultiModalData",
    "MultiModalPlugin",
    "MULTIMODAL_REGISTRY",
    "MultiModalRegistry",
    "EXTERNAL_MM_DATA_TYPE",
]
