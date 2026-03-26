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
Preprocess the QA dataset to parquet format
"""

import re
import os
import random
import datasets
from pathlib import Path

from verl.utils.hdfs_io import copy, makedirs
import argparse

from prompt_pool import PROMPT_TEMPLATE_QWEN, PROMPT_TEMPLATE_LLAMA

random.seed(42)


def make_prefix(dp, llm):
    question = dp['question']
    if llm == "qwen":
        prefix = PROMPT_TEMPLATE_QWEN.format_map({"question": question})
    elif llm == "llama":
        prefix = PROMPT_TEMPLATE_LLAMA.format_map({"question": question})
    else:
        raise NotImplementedError

    return prefix


# python data_process/qa_test_gen.py --data_sources nq --model qwen
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_sources', default='nq')
    parser.add_argument('--model', type=str, default='qwen')

    args = parser.parse_args()
    model_name = args.model.lower()
    data_sources = args.data_sources.split(',')
    all_dataset = []

    folder_path = Path(args.local_dir).mkdir(parents=True, exist_ok=True)

    for data_source in data_sources:
        dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)
        if 'test' in dataset:
            print(f'Using the {data_source} test dataset...')
            test_dataset = dataset['test']
        elif 'dev' in dataset:
            print(f'Using the {data_source} dev dataset...')
            test_dataset = dataset['dev']
        else:
            print(f'Using the {data_source} train dataset...')
            test_dataset = dataset['train']

        # random sample
        test_dataset = test_dataset.select(random.sample(list(range(len(test_dataset))), min(len(test_dataset), 500)))

        # add a row to each data item that represents a unique id
        def make_map_fn(split):

            def process_fn(example, idx):
                example['question'] = example['question'].strip()
                if example['question'][-1] != '?':
                    example['question'] += '?'
                question = make_prefix(example, llm=model_name)
                solution = {
                    "target": example['golden_answers'],
                }

                data = {
                    "data_source": data_source,
                    "prompt": [{
                        "role": "user",
                        "content": question,
                    }],
                    "ability": "fact-reasoning",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": solution
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
                return data

            return process_fn

        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
        all_dataset.append(test_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    all_test_dataset = datasets.concatenate_datasets(all_dataset)
    all_test_dataset.to_parquet(os.path.join(local_dir, 'test_{}_{}.parquet'.format(data_source, model_name)))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
