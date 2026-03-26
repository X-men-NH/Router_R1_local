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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import os
import re
import numpy as np
from collections import deque
import pickle

from router_r1.llm_agent.route_service import check_llm_name


PUNISH_REWARD_MAX = -1.0
PUNISH_REWARD_MEDIUM = -1.0
PUNISH_REWARD_SMALL = -1.0


window_size = 1000
use_percentile = True
q_low, q_high = 0.05, 0.95
eps = 1e-8

transform = 'sqrt'
alpha = 0.01 

buffer = deque(maxlen=window_size)


def reward_preprocess(r: float) -> float:
    if transform == 'log':
        return np.log1p(alpha * r)
    elif transform == 'sqrt':
        return np.sqrt(r)
    else:
        return r


def normalize_reward(r):
    r = reward_preprocess(r)
    buffer.append(r)
    arr = np.array(buffer)
    if use_percentile and len(arr) >= 2:
        r_min = np.percentile(arr, 100 * q_low)
        r_max = np.percentile(arr, 100 * q_high)
    else:
        r_min = arr.min()
        r_max = arr.max()

    denom = r_max - r_min
    if denom < eps:
        return 0.5

    r_scaled = (r - r_min) / denom
    return 1.0 - float(np.clip(r_scaled, 0.0, 1.0))


def is_valid_llm_name(target_llm):
    target_llm = target_llm.strip().lower()
    LLM_NAME, _ = check_llm_name(target_llm=target_llm)
    if LLM_NAME == "":
        return False
    else:
        return True


def format_reward(completion):
    tag_enclose_pattern = r'<(search|answer|think|information)>(.*?)</\1>'
    tag_enclose_matches = re.findall(tag_enclose_pattern, completion, re.DOTALL)
    if len(tag_enclose_matches) == 0:
        return PUNISH_REWARD_MAX
    
    if completion.count("<search>") != completion.count("</search>") or completion.count("<think>") != completion.count("</think>") or completion.count("<answer>") != completion.count("</answer>") or completion.count("<information>") != completion.count("</information>"):
        return PUNISH_REWARD_MAX

    route_enclose_count = 0
    answer_enclose_count = 0
    think_enclose_count = 0
    info_enclose_count = 0
    is_nesting = False
    query_format_punish = False
    llm_name_punish = False
    think_punish = False
    for single_match in tag_enclose_matches:
        action = single_match[0].strip()
        content = single_match[1].strip()
        if action == "search":
            route_enclose_count += 1
            if content.count(":") == 1:
                if content.split(":")[-1].strip() == '' or "llm-name" in content.strip().lower() \
                        or "your-query" in content.strip().lower() or content.split(":")[0].strip().lower() in content.split(":")[-1].strip().lower():
                    query_format_punish = True
                if not is_valid_llm_name(content.split(":")[0]):
                    llm_name_punish = True
            else:
                query_format_punish = True
        elif action == "answer":
            answer_enclose_count += 1
        elif action == "think":
            think_enclose_count += 1
            if content == "..." or content == "":
                think_punish = True
        else:
            info_enclose_count += 1
        
        if content.count("<search>") + content.count("</search>") + content.count("<think>") + content.count("</think>") + content.count("<answer>") + content.count("</answer>") + content.count("<information>") + content.count("</information>") != 0:
            is_nesting = True
    

    if think_punish:
        return PUNISH_REWARD_MAX

    if is_nesting:
        return PUNISH_REWARD_MAX
        
    if answer_enclose_count != 1 or think_enclose_count == 0 or route_enclose_count != info_enclose_count:
        return PUNISH_REWARD_MAX

    completion = completion.strip()
    if not completion[:len("<think>")] == "<think>":
        return PUNISH_REWARD_MAX

    if not completion.rfind("</answer>") + 9 == len(completion):
        return PUNISH_REWARD_MAX

    if query_format_punish:
        return PUNISH_REWARD_MEDIUM

    if llm_name_punish:
        return PUNISH_REWARD_SMALL

    return 0.0


def route_count(completion):
    tag_enclose_pattern = r'<(search|answer)>(.*?)</\1>'
    tag_enclose_matches = re.findall(tag_enclose_pattern, completion, re.DOTALL)
    if len(tag_enclose_matches) == 0:
        return 0

    route_enclose_count = 0
    query_format_punish = False
    llm_name_punish = False
    for single_match in tag_enclose_matches:
        action = single_match[0].strip()
        content = single_match[1].strip()
        if action == "search":
            if content.count(":") == 1:
                if content.split(":")[
                    -1].strip() == '' or "llm-name" in content.strip().lower() or "your-query" in content.strip().lower():
                    query_format_punish = True
                if not is_valid_llm_name(content.split(":")[0]):
                    llm_name_punish = True
            else:
                query_format_punish = True

            if not query_format_punish and not llm_name_punish:
                route_enclose_count += 1

    return route_enclose_count


def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, config, tokenizer, num_examine, format_score=0., state="train", reward_metric="f1", max_turns=4, max_obs_length=512, cost_coe=0.0) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.state = state
        self.reward_metric = reward_metric
        self.max_turns = max_turns
        self.max_obs_length = max_obs_length
        self.cost_coe = cost_coe

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        cost_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        metric_em_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        metric_f1_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        metric_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        route_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        token_price = np.array(data.meta_info['batch_completion_tokens'])
        normalized_cost_reward = []
        already_print_data_sources = {}
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            # sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences = valid_response_ids
            sequences_str = self.tokenizer.decode(sequences)
            strict_format_score = format_reward(completion=sequences_str)
            route_cnt = route_count(completion=sequences_str)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # API Cost Reward
            api_cost = normalize_reward(token_price[i])
            normalized_cost_reward.append(api_cost)

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            if self.state == "train":
                metric_score, cost_score, reward_score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=strict_format_score, api_cost=api_cost, state=self.state, reward_metric=self.reward_metric, cost_coe=self.cost_coe)
                metric_tensor[i, valid_response_length - 1] = metric_score
            else:
                metric_score_em, metric_score_f1, cost_score, reward_score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=strict_format_score, api_cost=api_cost, state=self.state, reward_metric=self.reward_metric, cost_coe=self.cost_coe)
                metric_em_tensor[i, valid_response_length - 1] = metric_score_em
                metric_f1_tensor[i, valid_response_length - 1] = metric_score_f1

            reward_tensor[i, valid_response_length - 1] = reward_score
            cost_tensor[i, valid_response_length - 1] = cost_score
            route_tensor[i, valid_response_length - 1] = route_cnt
            # all_scores.append(score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                # print(sequences_str)
        
        # print(f"[DEBUG] all_scores: {all_scores}")
        # print(f"[DEBUG] all_scores shape: {np.array(all_scores).shape}")
        # print(f"[DEBUG] all_scores mean: {np.mean(all_scores)}")
        # print(f"[DEBUG] all_scores max: {np.max(all_scores)}")
        # print(f"[DEBUG] all_scores min: {np.min(all_scores)}")
        # print(f"[DEBUG] all_scores std: {np.std(all_scores)}")
        # print(normalized_cost_reward)

        if self.state == "train":
            return metric_tensor, cost_tensor, reward_tensor
        else:
            return metric_em_tensor, metric_f1_tensor, cost_tensor, reward_tensor, route_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(config=config, tokenizer=tokenizer, num_examine=0, state="train", reward_metric=config.reward_metric, max_turns=config.max_turns, max_obs_length=config.data.max_obs_length, cost_coe=config.cost_coe)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(config=config, tokenizer=tokenizer, num_examine=1, state="val", reward_metric=config.reward_metric, max_turns=config.max_turns, max_obs_length=config.data.max_obs_length, cost_coe=config.cost_coe)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
