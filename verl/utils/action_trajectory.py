from __future__ import annotations

import torch


def get_response_info_mask(full_info_mask: torch.Tensor, prompt_length: int, response_length: int) -> torch.Tensor:
    return full_info_mask[prompt_length:prompt_length + response_length]


def extract_action_response_ids(response_ids: torch.Tensor, response_info_mask: torch.Tensor) -> torch.Tensor:
    if response_ids.dim != 1:
        response_ids = response_ids.reshape(-1)
    if response_info_mask.dim != 1:
        response_info_mask = response_info_mask.reshape(-1)

    action_mask = response_info_mask.to(dtype=torch.bool)
    return response_ids[action_mask]


def extract_action_response_text(tokenizer, response_ids: torch.Tensor, response_info_mask: torch.Tensor) -> str:
    action_ids = extract_action_response_ids(response_ids, response_info_mask)
    if action_ids.numel() == 0:
        return ''

    text = tokenizer.decode(action_ids, skip_special_tokens=False)
    pad_token = getattr(tokenizer, 'pad_token', None)
    if pad_token:
        text = text.replace(pad_token, '')
    return text.strip()


def get_last_action_token_index(response_attention_mask: torch.Tensor, response_info_mask: torch.Tensor) -> int:
    if response_info_mask.dim != 1:
        response_info_mask = response_info_mask.reshape(-1)
    if response_attention_mask.dim != 1:
        response_attention_mask = response_attention_mask.reshape(-1)

    action_positions = torch.nonzero(response_info_mask.to(dtype=torch.bool), as_tuple=False).flatten()
    if action_positions.numel() > 0:
        return int(action_positions[-1].item())

    valid_positions = torch.nonzero(response_attention_mask.to(dtype=torch.bool), as_tuple=False).flatten()
    if valid_positions.numel() > 0:
        return int(valid_positions[-1].item())

    return max(int(response_attention_mask.shape[0]) - 1, 0)