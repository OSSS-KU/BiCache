from typing import (
    Dict,
    List,
    Tuple,
    Optional
)
from itertools import takewhile

from datasets import load_dataset
import torch
from torch.nn.functional import cosine_similarity
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from tqdm import tqdm


class ProfilerBase:
    def __init__(
        self,
        device: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer

    
    def tokenize(self, sequence: List[Dict[str, str]]) -> Tuple[List[int], List[int]]:
        raise NotImplementedError


    def generate_kvs(self, prefix_ids: List[int], user_prompt_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


    def calculate_cosine_similarity(self, kv1, kv2) -> torch.Tensor:
        res = []
        for (k1, v1), (k2, v2) in zip(kv1, kv2):
            sim_k = cosine_similarity(k1, k2, dim=1)
            sim_v = cosine_similarity(v1, v2, dim=1)
            res.append(torch.mean(torch.stack((sim_k, sim_v))))
        return torch.tensor(res).to(self.device)
    

    def set_number_of_inter_request_caching_layer(self, similarity: List[torch.Tensor], threshold: float) -> Dict[int, int]:
        number_of_inter_request_caching_layer = {}

        previous = 0
        for prefix_ratio, s in enumerate(similarity):
            n = sum(1 for _ in takewhile(lambda x: x > threshold, s))
            if previous < n:
                previous = n
                number_of_inter_request_caching_layer[prefix_ratio] = n
        return number_of_inter_request_caching_layer
    

    def profile(
        self,
        dataset_name: str,
        ids: List[List[Tuple[str, int]]],
        max_sequence_length: Optional[int] = None,
        num_profiling_data_per_ratio: int = 100,
        threshold: float = 0.95
    ) -> Dict[int, int]:
        ds = load_dataset(dataset_name, split="train")
        datas = []

        for r, l in enumerate(tqdm(ids, desc="profiling")):
            if len(l) < num_profiling_data_per_ratio:
                raise ValueError("Not enough data exist.")
            
            d = []
            for id, n_t in l:
                if len(d) == num_profiling_data_per_ratio:
                    break
                conversation = ds['conversation'][id][:n_t+1]
                sequence = [{"role": c["role"], "content": c['content']} for c in conversation]

                prefix_ids, user_prompt_ids = self.tokenize(sequence)
                prefix_len = len(prefix_ids)
                seq_len = prefix_len + len(user_prompt_ids)
                prefix_ratio = round(prefix_len / seq_len * 100)
                assert prefix_ratio == r

                if max_sequence_length is not None and seq_len > max_sequence_length:
                    continue

                kv1, kv2 = self.generate_kvs(prefix_ids, user_prompt_ids) # [(k1, v1), ...], [(k2, v2), ...]
                res = self.calculate_cosine_similarity(kv1, kv2)
                d.append(res)
            datas.append(d)

        for d in datas:
            if len(d) < num_profiling_data_per_ratio:
                raise ValueError("Not enough data exist.")
            
        return self.set_number_of_inter_request_caching_layer([torch.mean(torch.stack(d), dim=0) for d in datas], threshold)
    

class LLaDAProfiler(ProfilerBase):
    def tokenize(self, sequence: List[Dict[str, str]]) -> Tuple[List[int], List[int]]:
        prefix_ids = self.tokenizer.apply_chat_template(sequence[:-1], add_generation_prompt=True, tokenize=True)[:-6]
        user_prompt_ids = self.tokenizer.apply_chat_template([sequence[-1]], add_generation_prompt=True, tokenize=True)[1:]

        return prefix_ids, user_prompt_ids


    def generate_kvs(self, prefix_ids: List[int], user_prompt_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        prefix_len = len(prefix_ids)
        prefix_ids = torch.tensor(prefix_ids, dtype=torch.long, device=self.device)
        sequence = torch.concat([prefix_ids, torch.tensor(user_prompt_ids, dtype=torch.long, device=self.device)])

        with torch.inference_mode():
            _, prefix_cache = self.model(prefix_ids.unsqueeze(0))
            _, cache = self.model(sequence.unsqueeze(0))

        return (
            [
            (k.squeeze(0).permute(1, 0, 2).contiguous().view(-1, 32 * 128), 
             v.squeeze(0).permute(1, 0, 2).contiguous().view(-1, 32 * 128)
            ) for k, v, _ in prefix_cache
            ], 
            [
            (k[:, :, :prefix_len, :].squeeze(0).permute(1, 0, 2).contiguous().view(-1, 32 * 128), 
             v[:, :, :prefix_len, :].squeeze(0).permute(1, 0, 2).contiguous().view(-1, 32 * 128)
            ) for k, v, _ in cache
            ], 
        )
