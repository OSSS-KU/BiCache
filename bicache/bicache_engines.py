from collections import OrderedDict
from typing import (
    Optional,
    Dict,
    List,
    Tuple,
)
from array import array

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from tqdm import tqdm
import time
import xxhash


class EngineBase:
    def __init__(
        self,
        device: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        number_of_inter_request_caching_layer: Dict[int, int],
        intra_request_cache_update_interval: Optional[int] = None,
        cache_budget: Optional[int] = None,
        show_speed: bool = False,
    ):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer

        self.number_of_inter_request_caching_layer = number_of_inter_request_caching_layer
        self.intra_request_cache_update_interval = intra_request_cache_update_interval if intra_request_cache_update_interval else 1

        self.inter_request_caches: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {} # prefix hash -> [(k, v, o), ]

        self.cache_budget = cache_budget
        self.num_cache_tokens = 0
        self.cache_access_history = OrderedDict()

        self.show_speed = show_speed
        self.data_movements = 0
        self.data_movement_overhead = 0.
        self.cache_misses = 0
        self.cache_miss_overhead = 0.
        
        self.ttft = None


    def warm_up(self, input_length: int, n_step: int):
        raise NotImplementedError


    def check_memory_capacity(self, prefix_len: int) -> bool:
        return prefix_len < self.cache_budget - self.num_cache_tokens


    def evict_prefix_cache(self):
        if len(self.cache_access_history) == 0:
            raise RuntimeError("prefix length is longer than cache budget")
        
        key = self.cache_access_history.popitem(last=False)[0]
        self.num_cache_tokens -= self.inter_request_caches[key][0][0].size(2)
        del self.inter_request_caches[key]


    def get_prefix_hash(self, prefix_ids: List[int]) -> int:
        return xxhash.xxh64_intdigest(array('I', prefix_ids))


    def generate_inter_request_cache(self, key: int, prefix_ids: List[int]) -> None:
        raise NotImplementedError
    

    def get_prefix_ratio(self, seq_len: int, prefix_len: int) -> int:
        return round(prefix_len / seq_len * 100)
    

    def get_number_of_inter_request_caching_layer(self, prefix_ratio: int) -> int:
        k = max([r for r in self.number_of_inter_request_caching_layer.keys() if r <= prefix_ratio])
        return self.number_of_inter_request_caching_layer[k]


    def get_inter_request_cache(self, prefix_ids: List[int], prefix_ratio: int) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]: # ([(k, v), ], o)
        key = self.get_prefix_hash(prefix_ids)
        number_of_caching_layers = self.get_number_of_inter_request_caching_layer(prefix_ratio)
        assert number_of_caching_layers > 0

        if key not in self.inter_request_caches:
            if self.show_speed:
                self.cache_misses += 1
                start_time = time.perf_counter()

            while not self.check_memory_capacity(len(prefix_ids)):
                self.evict_prefix_cache()
            self.num_cache_tokens += len(prefix_ids)

            self.generate_inter_request_cache(key, prefix_ids)

            if self.show_speed:
                self.cache_miss_overhead += time.perf_counter() - start_time

        if key in self.cache_access_history:
            self.cache_access_history.move_to_end(key)
        else:
            self.cache_access_history[key] = None
        cache = [(c[0], c[1]) for c in self.inter_request_caches[key][:number_of_caching_layers]]
        o = self.inter_request_caches[key][number_of_caching_layers-1][2]

        return (
            cache,
            o
        )


    def tokenize(self, sequence: List[Dict]) -> Tuple[List[int], int]: # (token ids, prefix length)
        raise NotImplementedError


    def generate(
        self, 
        sequence: List[Dict],
        steps: int,
        gen_length: int,
        mask_id: int,
        temperature: float,
    ) -> torch.Tensor:
        raise NotImplementedError


class LLaDAEngine(EngineBase):
    def add_gumbel_noise(self, logits, temperature):
        '''
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
        '''
        if temperature == 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (- torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise


    def get_num_transfer_tokens(self, mask_index, steps):
        '''
        In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
        Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
        the expected number of tokens transitioned at each step should be consistent.

        This function is designed to precompute the number of tokens that need to be transitioned at each step.
        '''
        mask_num = mask_index.sum(dim=1, keepdim=True)

        base = mask_num // steps
        remainder = mask_num % steps

        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, :remainder[i]] += 1

        return num_transfer_tokens
    

    def warm_up(self, input_length: int, n_step: int):
        for _ in tqdm(range(n_step), desc="Warm-up"):
            input_ids = torch.randint(
                low=0,
                high=self.tokenizer.vocab_size,
                size=(1, input_length),
                device=self.device,
                dtype=torch.long,
            )
            _ = self.model(input_ids)


    def generate_inter_request_cache(self, key: int, prefix_ids: List[int]) -> None:
        with torch.inference_mode():
            _, cache = self.model(torch.tensor(prefix_ids, dtype=torch.long, device=self.device).unsqueeze(0))
            self.inter_request_caches[key] = [(k, v, o) for k, v, o in cache]
        

    def tokenize(self, sequence: List[Dict]) -> Tuple[List[int], int]:
        assert sequence[-1]["role"] == "user"
        
        prefix = sequence[:-1]
        if len(prefix) == 0:
            return self.tokenizer.apply_chat_template(sequence, add_generation_prompt=True, tokenize=True), 0
        
        prefix_ids = self.tokenizer.apply_chat_template(prefix, add_generation_prompt=True, tokenize=True)[:-6]
        user_prompt_ids = self.tokenizer.apply_chat_template([sequence[-1]], add_generation_prompt=True, tokenize=True)[1:]

        return prefix_ids + user_prompt_ids, len(prefix_ids)


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

        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        x = torch.full((input_ids.shape[0], input_ids.shape[1] + gen_length), mask_id, dtype=torch.long).to(self.device)
        x[:, :input_ids.shape[1]] = input_ids.clone()

        mask_index = (x[:, input_ids.shape[1]:] == mask_id)
        num_transfer_tokens = self.get_num_transfer_tokens(mask_index, steps)

        if self.show_speed:
            start_time = time.perf_counter()
        with torch.inference_mode():
            for i in range(steps):
                if i % self.intra_request_cache_update_interval == 0:
                    prefix_cache, prefix_hidden_state = inter_request_cache

                mask_index = (x[:, prefix_len:] == mask_id)
                logits, kvo = self.model(
                    x[:, prefix_len:],
                    prefix_len=prefix_len,
                    prefix_cache=prefix_cache,
                    prefix_hidden_state=prefix_hidden_state,
                )
                logits = logits.logits[:, prefix_len:]

                if i % self.intra_request_cache_update_interval == 0:
                    prefix_cache = [(k[:, :, :prefix_len, :], v[:, :, :prefix_len, :]) for k, v, _ in kvo]
                    prefix_hidden_state = kvo[-1][2][:, :prefix_len, :] if kvo[-1][2] is not None else prefix_hidden_state

                logits_with_noise = self.add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l

                x0 = torch.where(mask_index, x0, x[:, prefix_len:])
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[:, prefix_len:][transfer_index] = x0[transfer_index]

                if i == 0 and self.show_speed:
                    self.ttft = time.perf_counter() - start_time

        return x[0][prefix_len:]

