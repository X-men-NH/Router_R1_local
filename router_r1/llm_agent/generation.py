import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
from .route_service import access_routing_pool, check_llm_name


ACTION_TAGS = ('decompose', 'search', 'subanswer', 'answer')
MAX_DECOMPOSE_QUESTIONS = 3
RESPONSE_TAGS = ('think',) + ACTION_TAGS
SUBQ_REF_RE = re.compile(r'^\s*\[SubQ(\d+)\]\s*(.*)$', re.IGNORECASE | re.DOTALL)


@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    exp_name: str = None
    api_base: str = None
    api_key: str = None

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

        self.log_questions = os.getenv('ROUTER_LOG_QUESTIONS', '0').lower() in ('1', 'true', 'yes', 'on')
        self.log_max_questions = int(os.getenv('ROUTER_LOG_MAX_QUESTIONS', '3'))
        self.log_max_chars = int(os.getenv('ROUTER_LOG_MAX_CHARS', '240'))
        self.log_route_trace = os.getenv('ROUTER_LOG_ROUTE_TRACE', '1').lower() in ('1', 'true', 'yes', 'on')
        self.log_max_events = int(os.getenv('ROUTER_LOG_MAX_EVENTS', '5'))
        self.skip_stalled_step = os.getenv('ROUTER_SKIP_STALLED_STEP', '1').lower() in ('1', 'true', 'yes', 'on')
        self.max_stalled_turns = int(os.getenv('ROUTER_MAX_STALLED_TURNS', '2'))

    def _clip_for_log(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = text.replace('\n', ' ').strip()
        if self.log_max_chars <= 0 or len(text) <= self.log_max_chars:
            return text
        return text[:self.log_max_chars] + '...'

    def _format_invalid_subanswer_feedback(self, state: Optional[List[Dict[str, Any]]]) -> str:
        state_block = self._format_decomposition_state(state)
        if state is None:
            reminder = (
                'No decomposition is active. <subanswer> is only valid after <decompose> creates a '
                '<decomposition_state> block. If this is a simple or single-hop question, provide the final '
                'result with <answer>...</answer> instead.'
            )
        elif self._get_current_subq(state) is None:
            reminder = 'All sub-questions are already DONE. Provide the final result with <answer>...</answer>.'
        else:
            current_subq = self._get_current_subq(state)
            reminder = (
                f'Use <subanswer>[SubQ{current_subq["id"]}] ... </subanswer> only for the current first TODO '
                'sub-question shown in <decomposition_state>.'
            )

        if state_block:
            return f"\n\n{state_block}\n{reminder}\n\n"
        return f"\n\n{reminder}\n\n"

    def _format_invalid_search_feedback(self, state: Optional[List[Dict[str, Any]]]) -> str:
        state_block = self._format_decomposition_state(state)
        if state is None:
            reminder = (
                'No decomposition is active. Use <search>ModelName:query</search> for a direct search or '
                '<answer>...</answer> if you already know the answer.'
            )
        elif self._get_current_subq(state) is None:
            reminder = 'All sub-questions are already DONE. Provide the final result with <answer>...</answer>.'
        else:
            current_subq = self._get_current_subq(state)
            reminder = (
                f'Use <search>[SubQ{current_subq["id"]}] ModelName:query</search> only for the current first TODO '
                'sub-question shown in <decomposition_state>.'
            )

        if state_block:
            return f"\n\n{state_block}\n{reminder}\n\n"
        return f"\n\n{reminder}\n\n"

    def _split_route_content(self, content: str) -> Tuple[str, str]:
        if not isinstance(content, str):
            return '', ''
        parts = content.split(':', 1)
        if len(parts) != 2:
            return content.strip(), ''
        return parts[0].strip(), parts[1].strip()

    def _extract_subq_reference(self, content: str) -> Tuple[Optional[int], str]:
        if not isinstance(content, str):
            return None, ''

        stripped = content.strip()
        match = SUBQ_REF_RE.match(stripped)
        if match is None:
            return None, stripped

        return int(match.group(1)), match.group(2).strip()

    def _find_subq(self, state: Optional[List[Dict[str, Any]]], subq_id: Optional[int]):
        if state is None or subq_id is None:
            return None
        return next((item for item in state if item.get('id') == subq_id), None)

    def _format_decomposition_guidance(self, state: Optional[List[Dict[str, Any]]]) -> str:
        current_subq = self._get_current_subq(state)
        if current_subq is None:
            return 'All sub-questions are DONE. Provide the final result with <answer>...</answer>.'
        return (
            f'For the current TODO, use <search>[SubQ{current_subq["id"]}] ModelName:query</search> and '
            f'<subanswer>[SubQ{current_subq["id"]}] ... </subanswer>.'
        )

    def _is_valid_llm_name(self, target_llm: str) -> bool:
        if not isinstance(target_llm, str):
            return False
        llm_name, _ = check_llm_name(target_llm=target_llm.strip().lower())
        return llm_name != ''

    def _parse_decomposition_items(self, content: str) -> List[str]:
        if not isinstance(content, str):
            return []

        items = []
        for raw_line in content.splitlines():
            line = re.sub(r'^\s*(?:[-*]|\d+[.)]?)\s*', '', raw_line).strip()
            if line:
                items.append(line)
        return items

    def _format_decomposition_state(self, state: List[Dict[str, Any]]) -> str:
        if not state:
            return ''

        lines = ['<decomposition_state>']
        for item in state:
            status = 'DONE' if item.get('done') else 'TODO'
            line = f"[SubQ{item['id']}][{status}] {item['question']}"
            if item.get('done') and item.get('answer'):
                answer = self._clip_for_log(item['answer'])
                line += f" => {answer}"
            lines.append(line)
        lines.append('</decomposition_state>')
        return '\n'.join(lines)

    def _get_current_subq(self, state: List[Dict[str, Any]]):
        if not state:
            return None
        return next((item for item in state if not item.get('done')), None)

    def _extract_question_text(self, prompt_text: str) -> str:
        if not isinstance(prompt_text, str):
            return ''

        question = prompt_text
        final_question_marker = '## Now answer the following question:'
        if final_question_marker in question:
            question = question.rsplit(final_question_marker, 1)[-1]

        line_matches = re.findall(r'(?m)^\s*Question:\s*(.+?)\s*$', question)
        if line_matches:
            question = line_matches[-1].strip()
        else:
            inline_matches = re.findall(r'Question:\s*([^\n]+)', question)
            if inline_matches:
                question = inline_matches[-1].strip()
            else:
                # Fallback for chat-template variants without explicit marker.
                question = prompt_text.split('user')[-1].strip() if 'user' in prompt_text else prompt_text.strip()

        # Trim possible assistant suffix introduced by chat templates.
        for stop_token in ['\nassistant', '<|im_start|>assistant', '<|start_header_id|>assistant']:
            pos = question.find(stop_token)
            if pos != -1:
                question = question[:pos].strip()

        return question

    def _maybe_log_route_trace(self,
                               step: int,
                               active_mask: torch.Tensor,
                               cur_actions: List[str],
                               contents: List[str],
                               dones: List[int],
                               question_texts: List[str]):
        if not self.log_route_trace:
            return

        active_indices = torch.nonzero(active_mask, as_tuple=False).flatten().tolist()
        if not active_indices:
            return

        sample_indices = active_indices[:self.log_max_events]
        print(f"[TRACE step={step}] events={len(sample_indices)}/{len(active_indices)}")
        for idx in sample_indices:
            action = cur_actions[idx] if idx < len(cur_actions) else None
            content = contents[idx] if idx < len(contents) else ''
            done = int(dones[idx]) if idx < len(dones) else -1
            question = self._clip_for_log(question_texts[idx]) if idx < len(question_texts) else ''

            if action == 'search':
                subq_id, content = self._extract_subq_reference(content)
                route_model, route_query = self._split_route_content(content)
                route_query = self._clip_for_log(route_query)
                subq_prefix = f" subq={subq_id}" if subq_id is not None else ''
                print(
                    f"[TRACE step={step}] idx={idx} action=search{subq_prefix} model={route_model!r} "
                    f"query={route_query!r} done={done} question={question!r}"
                )
            elif action == 'answer':
                answer_preview = self._clip_for_log(content)
                print(
                    f"[TRACE step={step}] idx={idx} action=answer done={done} "
                    f"answer={answer_preview!r} question={question!r}"
                )
            elif action == 'decompose':
                plan_preview = self._clip_for_log(content).replace('\n', ' | ').strip()
                print(
                    f"[TRACE step={step}] idx={idx} action=decompose done={done} "
                    f"plan={plan_preview!r} question={question!r}"
                )
            elif action == 'subanswer':
                subq_id, content = self._extract_subq_reference(content)
                subanswer_preview = self._clip_for_log(content)
                subq_prefix = f" subq={subq_id}" if subq_id is not None else ''
                print(
                    f"[TRACE step={step}] idx={idx} action=subanswer{subq_prefix} done={done} "
                    f"subanswer={subanswer_preview!r} question={question!r}"
                )
            else:
                print(
                    f"[TRACE step={step}] idx={idx} action={action!r} done={done} question={question!r}"
                )

    def _maybe_log_active_questions(self, step: int, active_mask: torch.Tensor, question_texts: List[str]):
        if not self.log_questions:
            return
        active_indices = torch.nonzero(active_mask, as_tuple=False).flatten().tolist()
        if not active_indices:
            print(f"[STEP {step}] active_questions=0")
            return

        sample_indices = active_indices[:self.log_max_questions]
        print(f"[STEP {step}] active_questions={len(active_indices)}, sample={len(sample_indices)}")
        for local_idx, q_idx in enumerate(sample_indices):
            question = self._clip_for_log(question_texts[q_idx])
            print(f"[STEP {step}] q{local_idx}: {question}")

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        processed_responses = []
        for resp in responses_str:
            first_structured_match = re.search(
                r'<(' + '|'.join(RESPONSE_TAGS) + r')>(.*?)</\1>',
                resp,
                re.DOTALL,
            )
            if first_structured_match is not None:
                resp = resp[first_structured_match.start():]

            close_positions = []
            for tag in ACTION_TAGS:
                close_tag = f'</{tag}>'
                pos = resp.find(close_tag)
                if pos != -1:
                    close_positions.append((pos, close_tag))

            if close_positions:
                pos, close_tag = min(close_positions, key=lambda item: item[0])
                processed_responses.append(resp[:pos + len(close_tag)])
            else:
                processed_responses.append(resp)

        responses_str = processed_responses


        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        raw_prompt_texts = self.tokenizer.batch_decode(initial_input_ids, skip_special_tokens=True)
        question_texts = [self._extract_question_text(text) for text in raw_prompt_texts]
        batch_completion_tokens = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.float32)
        decomposition_states = [None for _ in range(gen_batch.batch['input_ids'].shape[0])]
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_route_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        prev_active_count = active_mask.sum().item()
        stalled_turns = 0
        stall_guard_triggered = False
        rollings = gen_batch
        meta_info = {}

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum().item() > 0:
                break
            if self.skip_stalled_step and stalled_turns >= self.max_stalled_turns:
                print(
                    f"[STALL_GUARD] step={step} no progress for {stalled_turns} turns; "
                    f"force-finishing remaining {active_mask.sum().item()} active samples"
                )
                active_mask = torch.zeros_like(active_mask, dtype=torch.bool)
                active_num_list.append(0)
                stall_guard_triggered = True
                break
            self._maybe_log_active_questions(step, active_mask, question_texts)
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            try:
                gen_output = self._generate_with_gpu_padding(rollings_active)
            except Exception as e:
                raise RuntimeError(f"Generation failed at turn {step} with active batch size {rollings_active.batch['input_ids'].shape[0]}") from e

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            cur_actions, contents = self.postprocess_predictions(responses_str)

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_route, cur_completion_tokens = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, decomposition_states=decomposition_states
            )
            self._maybe_log_route_trace(step, active_mask, cur_actions, contents, dones, question_texts)
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            current_active_count = active_mask.sum().item()
            active_num_list.append(current_active_count)
            if current_active_count > 0 and current_active_count == prev_active_count:
                stalled_turns += 1
            else:
                stalled_turns = 0
            prev_active_count = current_active_count
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_route_stats += torch.tensor(is_route, dtype=torch.int)
            batch_completion_tokens += torch.tensor(cur_completion_tokens, dtype=torch.float32)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )

        # final LLM rollout
        if active_mask.sum().item() > 0 and not stall_guard_triggered:
            self._maybe_log_active_questions(self.config.max_turns, active_mask, question_texts)
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            cur_actions, contents = self.postprocess_predictions(responses_str)

            # # Execute in environment and process observations
            _, dones, valid_action, is_route, cur_completion_tokens = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_route=False, decomposition_states=decomposition_states
            )
            self._maybe_log_route_trace(self.config.max_turns, active_mask, cur_actions, contents, dones, question_texts)

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_route_stats += torch.tensor(is_route, dtype=torch.int)
            batch_completion_tokens += torch.tensor(cur_completion_tokens, dtype=torch.float32)

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_route_stats'] = valid_route_stats.tolist()
        meta_info['batch_completion_tokens'] = batch_completion_tokens.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_route=True, decomposition_states=None) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_route, cur_completion_tokens = [], [], [], [], []
        if decomposition_states is None:
            decomposition_states = [None for _ in range(len(cur_actions))]
        
        route_queries = []
        for action, content in zip(cur_actions, contents):
            if action == 'search':
                _, route_content = self._extract_subq_reference(content)
                route_queries.append(route_content)
        if do_route:
            route_results, completion_tokens_list = self.batch_route(route_queries)
            assert len(route_results) == sum([1 for action in cur_actions if action == 'search'])
            assert len(route_results) == len(completion_tokens_list)
        else:
            route_results = [''] * sum([1 for action in cur_actions if action == 'search'])
            completion_tokens_list = [0.0] * sum([1 for action in cur_actions if action == 'search'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_route.append(0)
                cur_completion_tokens.append(0.0)
            else:
                if action == 'answer':
                    curr_state = decomposition_states[i]
                    unfinished = curr_state and self._get_current_subq(curr_state) is not None
                    if unfinished:
                        state_block = self._format_decomposition_state(curr_state)
                        current_subq = self._get_current_subq(curr_state)
                        reminder = (
                            f'Finish the current TODO sub-question with '
                            f'<subanswer>[SubQ{current_subq["id"]}] ... </subanswer> before giving the final answer.'
                        )
                        next_obs.append(f"\n\n{state_block}\n{reminder}\n\n")
                        dones.append(0)
                        valid_action.append(0)
                    else:
                        next_obs.append('')
                        dones.append(1)
                        valid_action.append(1)
                    is_route.append(0)
                    cur_completion_tokens.append(0.0)
                elif action == 'decompose':
                    plan_lines = self._parse_decomposition_items(contents[i])
                    if 2 <= len(plan_lines) <= MAX_DECOMPOSE_QUESTIONS:
                        decomposition_states[i] = [
                            {
                                'id': idx + 1,
                                'question': question,
                                'done': False,
                                'attempts': 0,
                                'answer': None,
                                'evidence': [],
                            }
                            for idx, question in enumerate(plan_lines)
                        ]
                        state_block = self._format_decomposition_state(decomposition_states[i])
                        guidance = self._format_decomposition_guidance(decomposition_states[i])
                        next_obs.append(f"\n\n{state_block}\n{guidance}\n\n")
                        valid_action.append(1)
                    else:
                        next_obs.append('')
                        valid_action.append(0)
                    dones.append(0)
                    is_route.append(0)
                    cur_completion_tokens.append(0.0)
                elif action == 'subanswer':
                    curr_state = decomposition_states[i]
                    current_subq = self._get_current_subq(curr_state)
                    subq_id, subanswer_text = self._extract_subq_reference(contents[i])
                    target_subq = self._find_subq(curr_state, subq_id)
                    if not subanswer_text:
                        next_obs.append(self._format_invalid_subanswer_feedback(curr_state))
                        valid_action.append(0)
                    elif current_subq is None:
                        next_obs.append(self._format_invalid_subanswer_feedback(curr_state))
                        valid_action.append(0)
                    elif subq_id is None or target_subq is None:
                        next_obs.append(self._format_invalid_subanswer_feedback(curr_state))
                        valid_action.append(0)
                    elif target_subq.get('done'):
                        next_obs.append(self._format_invalid_subanswer_feedback(curr_state))
                        valid_action.append(0)
                    elif target_subq['id'] != current_subq['id']:
                        next_obs.append(self._format_invalid_subanswer_feedback(curr_state))
                        valid_action.append(0)
                    else:
                        target_subq['answer'] = subanswer_text
                        target_subq['done'] = True
                        state_block = self._format_decomposition_state(curr_state)
                        guidance = self._format_decomposition_guidance(curr_state)
                        next_obs.append(f"\n\n{state_block}\n{guidance}\n\n")
                        valid_action.append(1)
                    dones.append(0)
                    is_route.append(0)
                    cur_completion_tokens.append(0.0)
                elif isinstance(action, str) and action.startswith('route invalid'):
                    next_obs.append(self._format_invalid_search_feedback(decomposition_states[i]))
                    dones.append(0)
                    valid_action.append(0)
                    is_route.append(0)
                    cur_completion_tokens.append(0.0)
                elif action == 'search':
                    curr_state = decomposition_states[i]
                    current_subq = self._get_current_subq(curr_state)
                    subq_id, _ = self._extract_subq_reference(contents[i])
                    target_subq = self._find_subq(curr_state, subq_id)
                    numbered_mode = curr_state is not None
                    numbering_invalid = False
                    if numbered_mode:
                        numbering_invalid = (
                            current_subq is None or
                            subq_id is None or
                            target_subq is None or
                            target_subq.get('done') or
                            subq_id != current_subq['id']
                        )
                    elif subq_id is not None:
                        numbering_invalid = True

                    if numbering_invalid:
                        route_results.pop(0)
                        completion_tokens_list.pop(0)
                        next_obs.append(self._format_invalid_search_feedback(curr_state))
                        dones.append(0)
                        valid_action.append(0)
                        is_route.append(0)
                        cur_completion_tokens.append(0.0)
                    elif route_results[0].strip().lower() == "llm name error":
                        state_block = self._format_decomposition_state(curr_state)
                        state_prefix = f"{state_block}\n" if state_block else ''
                        info_block = f"<information>{'[SubQ' + str(current_subq['id']) + '] ' if current_subq else ''}None</information>"
                        next_obs.append(f"\n\n{state_prefix}{info_block}\n\n")
                        route_results.pop(0)
                        valid_action.append(0)
                    elif route_results[0].strip().lower() == "api request error":
                        state_block = self._format_decomposition_state(curr_state)
                        state_prefix = f"{state_block}\n" if state_block else ''
                        info_block = f"<information>{'[SubQ' + str(current_subq['id']) + '] ' if current_subq else ''}None</information>"
                        next_obs.append(f"\n\n{state_prefix}{info_block}\n\n")
                        route_results.pop(0)
                        valid_action.append(0)
                    else:
                        routed_info = route_results.pop(0).strip()
                        if current_subq is not None:
                            current_subq['attempts'] = current_subq.get('attempts', 0) + 1
                            current_subq.setdefault('evidence', []).append(routed_info)
                            routed_info = f"[SubQ{current_subq['id']}] {routed_info}"
                        state_block = self._format_decomposition_state(curr_state)
                        state_prefix = f"{state_block}\n" if state_block else ''
                        next_obs.append(f"\n\n{state_prefix}<information>{routed_info}</information>\n\n")
                        valid_action.append(1)
                    dones.append(0)
                    is_route.append(1)
                    cur_completion_tokens.append(completion_tokens_list.pop(0))
                else:
                    next_obs.append('')
                    dones.append(0)
                    valid_action.append(0)
                    is_route.append(0)
                    cur_completion_tokens.append(0.0)
            
        assert len(route_results) == 0
        assert len(completion_tokens_list) == 0
            
        return next_obs, dones, valid_action, is_route, cur_completion_tokens

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(decompose|search|subanswer|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                    if action == "search" and ("llm-name" in content.strip().lower() or "your-query" in content.strip().lower()):
                        action = "route invalid-1"
                    elif action == "search":
                        _, route_content = self._extract_subq_reference(content)
                        if ":" not in route_content:
                            action = "route invalid-2"
                        elif route_content.strip().lower().split(":")[-1].strip() == "":
                            action = "route invalid-3"
                        else:
                            route_model, _ = self._split_route_content(route_content)
                            if not self._is_valid_llm_name(route_model):
                                action = "route invalid-4"
                    elif action == "decompose":
                        plan_lines = [line.strip(" -1234567890.") for line in content.splitlines() if line.strip()]
                        if len(plan_lines) < 2:
                            action = "decompose invalid-1"
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_route(self, queries: List[str] = None) -> str:
        ret = access_routing_pool(queries=queries, api_base=self.config.api_base, api_key=self.config.api_key)
        
        return ret['result'], ret["completion_tokens_list"]
