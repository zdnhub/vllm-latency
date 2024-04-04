"""A Neuron worker class."""
from typing import List, Optional

import torch
import torch.distributed

from vllm.config import (DeviceConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, CacheConfig)
from vllm.model_executor import set_random_seed
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.neuron_model_runner import NeuronModelRunner
from vllm.worker.worker_base import LoraNotSupportedWorkerBase


class NeuronWorker(LoraNotSupportedWorkerBase):
    """A worker class that executes the model on a group of neuron cores.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config

        self.model_runner = NeuronModelRunner(model_config, parallel_config,
                                              scheduler_config, device_config)

    def init_device(self) -> None:
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    # TODO change name
    def profile_num_available_blocks(self) -> tuple[int, int]:
        # Set the number of GPU blocks to be the same as the maximum number of
        # sequences that can be processed in a single batch. This is equivalent
        # to schedule without PagedAttention.
        num_gpu_blocks = self.scheduler_config.max_num_seqs

        # Swap not yet supported with Neuron backend.
        num_cpu_blocks = 0

        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        assert num_cpu_blocks == 0
        assert num_gpu_blocks == self.scheduler_config.max_num_seqs
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Optional[SamplerOutput]:
        num_seq_groups = len(seq_group_metadata_list)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return {}

        output = self.model_runner.execute_model(seq_group_metadata_list)
        return output
