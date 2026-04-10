# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, ShardedStateDictConfig, StateDictType, FullStateDictConfig
from torch.distributed.device_mesh import DeviceMesh

from verl.third_party.vllm import LLM
from verl.third_party.vllm import parallel_state as vllm_ps
from verl import DataProto
from verl.utils.torch_functional import (broadcast_dict_tensor, allgather_dict_tensors)
from verl.utils.debug import log_gpu_memory_usage

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


class FSDPVLLMShardingManager(BaseShardingManager):

    def __init__(self,
                 module: FSDP,
                 inference_engine: LLM,
                 model_config,
                 full_params: bool = False,
                 device_mesh: DeviceMesh = None):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh

        # Full params
        self.full_params = full_params
        if full_params:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.FULL_STATE_DICT,
                                     state_dict_config=FullStateDictConfig())
        else:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.SHARDED_STATE_DICT,
                                     state_dict_config=ShardedStateDictConfig())

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh['dp'].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    @staticmethod
    def _reset_fsdp_handles_to_idle(module):
        """Reset any FSDP handles stuck in FORWARD state back to IDLE.

        This can happen during multi-turn generation when a previous turn's
        forward pass didn't cleanly finish before the next rollout call.
        Only called as a fallback when the normal state_dict() path fails.
        """
        from torch.distributed.fsdp._common_utils import HandleTrainingState
        for m in module.modules():
            if isinstance(m, FSDP):
                handle = getattr(m, '_handle', None)
                if handle is None and hasattr(m, '_handles') and m._handles:
                    handle = m._handles[0]
                if handle is not None and hasattr(handle, '_training_state'):
                    if handle._training_state != HandleTrainingState.IDLE:
                        logger.warning(
                            f"Resetting FSDP handle from {handle._training_state} to IDLE"
                        )
                        handle._training_state = HandleTrainingState.IDLE

    @staticmethod
    def _clear_fsdp_unshard_context(module):
        """Best-effort cleanup for stale FSDP unshard contexts.

        Some multi-turn generation failures leave `_unshard_params_ctx` populated,
        which makes the next `state_dict()` assert even after handles are reset to
        IDLE. We clear these stale entries before retrying.
        """
        cleared = False
        for m in module.modules():
            if not isinstance(m, FSDP):
                continue

            for owner in (m, getattr(m, '_fsdp_state', None)):
                if owner is None:
                    continue
                ctx = getattr(owner, '_unshard_params_ctx', None)
                if isinstance(ctx, dict):
                    for key in list(ctx.keys()):
                        ctx[key] = None
                    cleared = True

        if cleared:
            logger.warning("Cleared stale FSDP _unshard_params_ctx entries before retrying state_dict().")

    @classmethod
    def _recover_fsdp_state_for_state_dict(cls, module):
        cls._reset_fsdp_handles_to_idle(module)
        cls._clear_fsdp_unshard_context(module)

    def __enter__(self):
        log_gpu_memory_usage('Before state_dict() in sharding manager memory', logger=logger)
        # Switch to eval mode before extracting state_dict.
        # In multi-turn generation (Router-R1), generate_sequences is called
        # multiple times per step.  Between turns the previous recompute_log_prob
        # forward pass may leave FSDP handles in FORWARD state.  eval() +
        # _no_grad context helps FSDP reset cleanly.  If state_dict() still
        # hits the FORWARD-state assertion, we fall back to resetting handles
        # via the private API (best-effort).
        self.module.eval()
        params = None
        recovery_attempts = 0
        while True:
            try:
                params = self.module.state_dict()
                break
            except AssertionError as e:
                error_text = str(e)
                recoverable = (
                    "HandleTrainingState.FORWARD" in error_text or
                    "_unshard_params_ctx[module] is not None" in error_text
                )
                if not recoverable or recovery_attempts >= 2:
                    raise

                recovery_attempts += 1
                logger.warning(
                    "FSDP state_dict() hit a recoverable multi-turn generation assertion; "
                    f"attempting recovery {recovery_attempts}/2. Error: {error_text}"
                )
                self._recover_fsdp_state_for_state_dict(self.module)
        log_gpu_memory_usage('After state_dict() in sharding manager memory', logger=logger)
        # Copy, not share memory
        load_format = 'hf' if self.full_params else 'dtensor'
        self.inference_engine.sync_model_weights(params, load_format=load_format)
        log_gpu_memory_usage('After sync model weights in sharding manager', logger=logger)

        del params
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After del state_dict and empty_cache in sharding manager', logger=logger)

        # TODO: offload FSDP model weights
        # self.module.cpu()
        # torch.cuda.empty_cache()
        # if torch.distributed.get_rank() == 0:
        # print(f'after model to cpu in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage('Before vllm offload in sharding manager', logger=logger)
        self.inference_engine.offload_model_weights()
        log_gpu_memory_usage('After vllm offload in sharding manager', logger=logger)

        # self.module.to('cuda')
        # if torch.distributed.get_rank() == 0:
        #     print(f'after actor module to cuda in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        data.batch = allgather_dict_tensors(data.batch.contiguous(),
                                            size=vllm_ps.get_tensor_model_parallel_world_size(),
                                            group=vllm_ps.get_tensor_model_parallel_group(),
                                            dim=0)

        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        broadcast_dict_tensor(data.batch,
                              src=vllm_ps.get_tensor_model_parallel_src_rank(),
                              group=vllm_ps.get_tensor_model_parallel_group())
        dp_rank = torch.distributed.get_rank()
        dp_size = torch.distributed.get_world_size()  # not consider torch micro-dp
        tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            # TODO: shall we build a micro_dp group for vllm when integrating with vLLM?
            local_prompts = data.chunk(chunks=tp_size)
            data = local_prompts[dp_rank % tp_size]
        return data
