import re
import argparse

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from data_process import prompt_pool
from router_r1.llm_agent.route_service import access_routing_pool


def get_query(text):
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    return matches[-1] if matches else None


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

    # Initial prompt
    prompt = prompt_pool.PROMPT_TEMPLATE_QWEN.format_map({"question": question})
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                               tokenize=False)

    # Sampling configuration
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1024,
        stop=["</search>", "</answer>"]
    )

    cnt = 0
    print('\n\n################# [Start Reasoning + Routing] ##################\n\n')
    STOP = False
    all_output = ""

    while True:
        if cnt > 4:
            break
        outputs = llm.generate(prompt, sampling_params=sampling_params)
        output_text = outputs[0].outputs[0].text
        if output_text.find("<answer>") != -1:
            STOP = True
            output_text += "</answer>"
        if not STOP:
            output_text += "</search>"

        print(f"[Generation {cnt}] Output:\n{output_text}")

        tmp_query = get_query(output_text)
        if tmp_query:
            route_results = route(tmp_query, api_base=api_base, api_key=api_key)
        else:
            route_results = ''

        if not STOP:
            prompt += curr_route_template.format(output_text=output_text, route_results=route_results)
            all_output += curr_route_template.format(output_text=output_text, route_results=route_results)
        else:
            all_output += output_text + "\n"
            break

        cnt += 1

    print('\n\n################# [Output] ##################\n\n')

    print(all_output)

    print('\n\n################# [Output] ##################\n\n')