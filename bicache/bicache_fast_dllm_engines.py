from collections import OrderedDict
from typing import (
    Optional,
    Dict,
    List,
    Tuple,
    Sequence,
)
from array import array

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from tqdm import tqdm
import time
import xxhash

from .bicache_engines import LLaDAEngine


class FastdLLMLLaDAEngine(LLaDAEngine):
    def __init__(
        self,
        device: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        number_of_inter_request_caching_layer: Dict[int, int],
        intra_request_cache_update_interval: Optional[int] = None,
        cache_budget: Optional[int] = None,
        show_speed: bool = False,
        block_length: int = 32,
    ):
        super().__init__(
            device=device,
            model=model,
            tokenizer=tokenizer,
            number_of_inter_request_caching_layer=number_of_inter_request_caching_layer,
            intra_request_cache_update_interval=intra_request_cache_update_interval,
            cache_budget=cache_budget,
            show_speed=show_speed
        )
        self.block_length = block_length

    
    def generate_dual_cache(
        self, 
        x: torch.Tensor, 
        s: int, 
        e: int, 
        prefix_len: int, 
        prefix_cache, 
        prefix_hidden_state,
    ) -> Tuple[Sequence[Tuple[torch.Tensor, torch.Tensor]], Sequence[Tuple[torch.Tensor, torch.Tensor]]]:
        _, cache = self.model(
            x[:, prefix_len:],
            prefix_len=prefix_len,
            prefix_cache=prefix_cache,
            prefix_hidden_state=prefix_hidden_state,
        )
        prefix = []
        suffix = []
        for k, v, _ in cache:
            prefix.append((k[:, :, prefix_len:s, :], v[:, :, prefix_len:s, :]))
            suffix.append((k[:, :, e:, :], v[:, :, e:, :]))
        return (prefix, suffix), cache


    def generate(
        self,
        sequence: List[Dict],
        steps: int = 128,
        gen_length: int = 128,
        mask_id: int = 126336,
        temperature: float = 0.,
    ) -> torch.Tensor:
        input_ids, prefix_len = self.tokenize(sequence)

        if prefix_len == 0:
            raise NotImplementedError
        
        prefix_ratio = self.get_prefix_ratio(len(input_ids)+gen_length, prefix_len)
        inter_request_cache = self.get_inter_request_cache(input_ids[:prefix_len], prefix_ratio)
        prefix_cache, prefix_hidden_state = inter_request_cache
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        x = torch.full((input_ids.shape[0], input_ids.shape[1] + gen_length), mask_id, dtype=torch.long).to(self.device)
        x[:, :input_ids.shape[1]] = input_ids.clone()

        assert gen_length % self.block_length == 0
        num_blocks = gen_length // self.block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        if self.show_speed:
            start_time = time.perf_counter()
        with torch.inference_mode():
            total_steps = 0
            for num_block in range(num_blocks):
                s = input_ids.shape[1] + num_block * self.block_length
                e = s + self.block_length

                block_mask_index = (x[:, s:e] == mask_id)
                num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps)
                for i in range(steps):
                    if total_steps % self.intra_request_cache_update_interval == 0:
                        prefix_cache, prefix_hidden_state = inter_request_cache
                    if i == 0:
                        dual_cache, kvo = self.generate_dual_cache(x, s, e, prefix_len, prefix_cache, prefix_hidden_state)
                        if total_steps % self.intra_request_cache_update_interval == 0:
                            prefix_cache = [(k[:, :, :prefix_len, :], v[:, :, :prefix_len, :]) for k, v, _ in kvo]
                            prefix_hidden_state = kvo[-1][2][:, :prefix_len, :] if kvo[-1][2] is not None else prefix_hidden_state

                    mask_index = (x[:, s:e] == mask_id)
                    logits, kvo = self.model(
                        x[:, s:e],
                        prefix_len=prefix_len,
                        prefix_cache=prefix_cache,
                        prefix_hidden_state=prefix_hidden_state,
                        dual_cache=dual_cache,
                    )
                    logits = logits.logits[:, prefix_len:]

                    if total_steps % self.intra_request_cache_update_interval == 0:
                        prefix_cache = [(k[:, :, :prefix_len, :], v[:, :, :prefix_len, :]) for k, v, _ in kvo]
                        prefix_hidden_state = kvo[-1][2][:, :prefix_len, :] if kvo[-1][2] is not None else prefix_hidden_state

                    logits_with_noise = self.add_gumbel_noise(logits, temperature=temperature)
                    x0 = torch.argmax(logits_with_noise, dim=-1)

                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

                    x0 = torch.where(mask_index, x0, x[:, s:e])
                    confidence = torch.where(mask_index, x0_p, -np.inf)

                    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                    for j in range(confidence.shape[0]):
                        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                        transfer_index[j, select_index] = True
                    x[:, s:e][transfer_index] = x0[transfer_index]

                    if total_steps == 0 and self.show_speed:
                        self.ttft = time.perf_counter() - start_time
                    total_steps += 1

        return x[0][prefix_len:]
    

    def generate_without_bicache(
        self, 
        sequence: List[Dict], 
        steps: int = 128, 
        gen_length: int = 128,
        mask_id: int = 126336,
        temperature: float = 0.,
    ) -> torch.Tensor:
        input_ids, prefix_len = self.tokenize(sequence)

        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        x = torch.full((input_ids.shape[0], input_ids.shape[1] + gen_length), mask_id, dtype=torch.long).to(self.device)
        x[:, :input_ids.shape[1]] = input_ids.clone()

        assert gen_length % self.block_length == 0
        num_blocks = gen_length // self.block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        if self.show_speed:
            start_time = time.perf_counter()
        with torch.inference_mode():
            total_steps = 0
            for num_block in range(num_blocks):
                s = input_ids.shape[1] + num_block * self.block_length
                e = s + self.block_length

                block_mask_index = (x[:, s:e] == mask_id)
                num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps)
                dual_cache, _ = self.generate_dual_cache(x, s, e, 0, None, None)
                dual_cache = ([(k[:, :, prefix_len:], v[:, :, prefix_len:]) for k, v in dual_cache[0]], dual_cache[1])

                for i in range(steps):
                    mask_index = (x[:, s:e] == mask_id)
                    logits, _ = self.model(
                        torch.cat((x[:, :prefix_len], x[:, s:e]), dim=1),
                        prefix_len=prefix_len,
                        dual_cache=dual_cache,
                    )
                    logits = logits.logits[:, prefix_len:]
                    logits_with_noise = self.add_gumbel_noise(logits, temperature=temperature)
                    x0 = torch.argmax(logits_with_noise, dim=-1)

                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

                    x0 = torch.where(mask_index, x0, x[:, s:e])
                    confidence = torch.where(mask_index, x0_p, -np.inf)
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                    for j in range(confidence.shape[0]):
                        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                        transfer_index[j, select_index] = True
                    x[:, s:e][transfer_index] = x0[transfer_index]

                    if total_steps == 0 and self.show_speed:
                        self.ttft = time.perf_counter() - start_time
                    total_steps += 1

        return x[0][prefix_len:]
