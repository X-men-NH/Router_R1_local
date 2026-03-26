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

import re
import string
import random
from collections import Counter

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 0:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def f1_score(prediction: str, ground_truth: str):
    """
    Calculate F1 score between prediction and ground truth.

    Args:
        prediction: Predicted text
        ground_truth: Ground truth text

    Returns:
        Tuple of (f1, precision, recall)
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = 0

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return ZERO_METRIC

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1., cost_coe=0.0, api_cost=0.0,
                     state="train", reward_metric="f1"):
    answer = extract_solution(solution_str=solution_str)
    if state == "train":
        do_print = random.randint(1, 64) == 1
        # do_print = True
    else:
        do_print = True

    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if state == "train":
        if answer is None:
            if format_score == -1.0:
                return 0, api_cost, format_score
            else:
                return 0, api_cost, format_score * (1.0 - cost_coe)
        else:
            if reward_metric == "f1":
                golden_answers = ground_truth['target']
                if isinstance(golden_answers, str):
                    golden_answers = [golden_answers]
                score = 0
                for golden_answer in golden_answers:
                    f1 = f1_score(answer, golden_answer)
                    if f1 > score:
                        score = f1

                if format_score == -1.0:
                    return score, api_cost, format_score
                else:
                    if score == 0:
                        return score, api_cost, score + format_score
                    else:
                        return score, api_cost, (score + format_score) * (1.0 - cost_coe) + api_cost * cost_coe
            else:
                if em_check(answer, ground_truth['target']):
                    if format_score == -1.0:
                        return score, api_cost, format_score
                    else:
                        return score, api_cost, (score + format_score) * (1.0 - cost_coe) + api_cost * cost_coe
                else:
                    if format_score == -1.0:
                        return format_score, api_cost, format_score
                    else:
                        return format_score, api_cost, format_score
    else:
        if answer is None:
            if format_score == -1.0:
                return 0, 0, api_cost, format_score
            else:
                return 0, 0, api_cost, format_score
        else:
            golden_answers = ground_truth['target']
            if isinstance(golden_answers, str):
                golden_answers = [golden_answers]
            score_f1 = 0
            for golden_answer in golden_answers:
                f1 = f1_score(answer, golden_answer)
                if f1 > score_f1:
                    score_f1 = f1

            if em_check(answer, ground_truth['target']):
                score_em = 1.0
            else:
                score_em = 0.0

            if format_score == -1.0:
                if reward_metric == "f1":
                    return score_em, score_f1, api_cost, format_score
                else:
                    return score_em, score_f1, api_cost, format_score
            else:
                if reward_metric == "f1":
                    if score_f1 == 0:
                        return score_em, score_f1, api_cost, score_f1 + format_score
                    else:
                        return score_em, score_f1, api_cost, (score_f1 + format_score) * (1.0 - cost_coe) + api_cost * cost_coe
                else:
                    if score_em == 0:
                        return score_em, score_f1, api_cost, score_em + format_score
                    else:
                        return score_em, score_f1, api_cost, (score_em + format_score) * (1.0 - cost_coe) + api_cost * cost_coe


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score
