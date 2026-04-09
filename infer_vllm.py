import re
import argparse

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from data_process import prompt_pool
from router_r1.llm_agent.route_service import access_routing_pool


ACTION_TAGS = ("decompose", "search", "subanswer", "answer")
MAX_DECOMPOSE_QUESTIONS = 3


def parse_action(text):
    matches = []
    for tag in ACTION_TAGS:
        pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
        for match in pattern.finditer(text):
            matches.append((match.start(), tag, match.group(1).strip()))

    if not matches:
        return None, None

    _, action, content = min(matches, key=lambda item: item[0])
    return action, content


def truncate_to_first_action(text):
    close_positions = []
    for tag in ACTION_TAGS:
        close_tag = f"</{tag}>"
        pos = text.find(close_tag)
        if pos != -1:
            close_positions.append((pos, close_tag))

    if not close_positions:
        return text

    pos, close_tag = min(close_positions, key=lambda item: item[0])
    return text[:pos + len(close_tag)]


def parse_decomposition_items(content):
    if not isinstance(content, str):
        return []

    items = []
    for raw_line in content.splitlines():
        line = re.sub(r'^\s*(?:[-*]|\d+[.)]?)\s*', '', raw_line).strip()
        if line:
            items.append(line)
    return items


def format_decomposition_state(state):
    if not state:
        return ''

    lines = ['<decomposition_state>']
    for item in state:
        status = 'DONE' if item.get('done') else 'TODO'
        line = f"[SubQ{item['id']}][{status}] {item['question']}"
        if item.get('done') and item.get('answer'):
            line += f" => {item['answer']}"
        lines.append(line)
    lines.append('</decomposition_state>')
    return '\n'.join(lines)


def format_invalid_subanswer_feedback(state):
    state_block = format_decomposition_state(state)
    if state is None:
        reminder = (
            'No decomposition is active. <subanswer> is only valid after <decompose> creates a '
            '<decomposition_state> block. If this is a simple or single-hop question, provide the final '
            'result with <answer>...</answer> instead.'
        )
    elif get_current_subq(state) is None:
        reminder = 'All sub-questions are already DONE. Provide the final result with <answer>...</answer>.'
    else:
        reminder = 'Use <subanswer> only to solve the current first TODO sub-question shown in <decomposition_state>.'

    if state_block:
        return f"\n{state_block}\n{reminder}\n"
    return f"\n{reminder}\n"


def get_current_subq(state):
    if not state:
        return None
    return next((item for item in state if not item.get('done')), None)


def route(query, api_base, api_key):
    ret = access_routing_pool(
        queries=[query],
        api_base=api_base,
        api_key=api_key
    )
    return ret['result'][0]


# NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=2,3,4,5 python infer_vllm.py
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', type=str, default="what are the countries of the united arab emirates?")
    parser.add_argument('--model_path', type=str, default="[YOUR_MODEL_PATH]")
    parser.add_argument('--api_base', type=str, default="[YOUR_API_BASE]")
    parser.add_argument('--api_key', type=str, default="[YOUR_API_KEY]")
    parser.add_argument('--max-turns', type=int, default=5)
    args = parser.parse_args()

    question = args.question
    model_id = args.model_path
    api_base = args.api_base
    api_key = args.api_key

    # Prepare the question
    question = question.strip()
    if question[-1] != '?':
        question += '?'

    # Model path and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = LLM(model=model_id, dtype="float16", tensor_parallel_size=torch.cuda.device_count())

    curr_route_template = '\n{output_text}\n<information>{route_results}</information>\n'
    curr_non_route_template = '\n{output_text}\n'
    decomposition_state = None

    # Initial prompt
    prompt = prompt_pool.PROMPT_TEMPLATE_QWEN.format_map({"question": question})
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                               tokenize=False)

    # Sampling configuration
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1024,
        stop=["</decompose>", "</search>", "</subanswer>", "</answer>"]
    )

    cnt = 0
    print('\n\n################# [Start Reasoning + Routing] ##################\n\n')
    STOP = False
    all_output = ""

    while True:
        if cnt >= args.max_turns:
            break
        outputs = llm.generate(prompt, sampling_params=sampling_params)
        output_text = truncate_to_first_action(outputs[0].outputs[0].text)
        action, content = parse_action(output_text)
        current_subq = get_current_subq(decomposition_state)
        if action == "answer" and current_subq is None:
            STOP = True

        print(f"[Generation {cnt}] Output:\n{output_text}")

        if action == "search" and content:
            route_results = route(content, api_base=api_base, api_key=api_key)
        else:
            route_results = ''

        if not STOP:
            if action == "answer" and current_subq is not None:
                stitched = (
                    f"\n{output_text}\n{format_decomposition_state(decomposition_state)}\n"
                    "Finish the current TODO sub-question with <subanswer> before giving the final answer.\n"
                )
                prompt += stitched
                all_output += stitched
            if action == "search":
                if decomposition_state:
                    current_subq = get_current_subq(decomposition_state)
                    if current_subq is not None:
                        current_subq['attempts'] = current_subq.get('attempts', 0) + 1
                        current_subq.setdefault('evidence', []).append(route_results)
                        route_results = f"[SubQ{current_subq['id']}] {route_results}"
                    state_block = format_decomposition_state(decomposition_state)
                    stitched = f"\n{output_text}\n{state_block}\n<information>{route_results}</information>\n"
                    prompt += stitched
                    all_output += stitched
                else:
                    prompt += curr_route_template.format(output_text=output_text, route_results=route_results)
                    all_output += curr_route_template.format(output_text=output_text, route_results=route_results)
            elif action == "subanswer":
                current_subq = get_current_subq(decomposition_state)
                if current_subq is not None and content:
                    current_subq["answer"] = content
                    current_subq["done"] = True
                    state_block = format_decomposition_state(decomposition_state)
                    stitched = f"\n{output_text}\n{state_block}\n"
                else:
                    stitched = f"\n{output_text}\n{format_invalid_subanswer_feedback(decomposition_state)}"
                prompt += stitched
                all_output += stitched
            elif action == "decompose":
                plan_lines = parse_decomposition_items(content)[:MAX_DECOMPOSE_QUESTIONS]
                decomposition_state = [
                    {"id": idx + 1, "question": question_text, "done": False, "attempts": 0, "answer": None, "evidence": []}
                    for idx, question_text in enumerate(plan_lines)
                ] if len(plan_lines) >= 2 else None
                state_block = format_decomposition_state(decomposition_state)
                stitched = f"\n{output_text}\n{state_block}\n" if state_block else curr_non_route_template.format(output_text=output_text)
                prompt += stitched
                all_output += stitched
            else:
                prompt += curr_non_route_template.format(output_text=output_text)
                all_output += curr_non_route_template.format(output_text=output_text)
        else:
            all_output += output_text + "\n"
            break

        cnt += 1

    print('\n\n################# [Output] ##################\n\n')

    print(all_output)

    print('\n\n################# [Output] ##################\n\n')