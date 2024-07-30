"""Token blocks."""
from typing import List, Optional

from vllm.utils import Device

DEFAULT_LAST_ACCESSED_TIME = -1


class PhysicalTokenBlock:
    """Represents the state of a block in the KV cache."""
    """modify:add"""

    def __init__(
        self,
        device: Device,
        block_number: int,
        block_size: int,
        block_hash: int,
        num_hashed_tokens: int,
        remote_rank: Optional[int] = 0,
    ) -> None:
        self.device = device
        self.block_number = block_number
        self.block_size = block_size
        self.block_hash = block_hash
        self.num_hashed_tokens = num_hashed_tokens

        self.ref_count = 0
        self.last_accessed = DEFAULT_LAST_ACCESSED_TIME

        self.computed = False

        self.remote_rank = remote_rank

    def __repr__(self) -> str:
        return (f'PhysicalTokenBlock(device={self.device}, '
                f'block_number={self.block_number}, '
                f'num_hashed_tokens={self.num_hashed_tokens}, '
                f'ref_count={self.ref_count}, '
                f'last_accessed={self.last_accessed}, '
                f'computed={self.computed},'
                f'remote_rank={self.remote_rank})')


# Mapping: logical block number -> physical block.
BlockTable = List[PhysicalTokenBlock]
