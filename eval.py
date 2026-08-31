import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from transformers import AutoTokenizer
import numpy as np
import json, re, time
from tqdm import tqdm

from model import LLaDAModelLM
from bicache import LLaDAProfiler, LLaDAEngine, FastdLLMLLaDAEngine
from system_prompts import get_system_prompt


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='GSAI-ML/LLaDA-8B-Instruct',
        mask_id=126336,
        steps=128,
        gen_length=128,
        block_length=32,
        device="cuda",
        save_path=None,
        show_speed=False,
        caching_policy=None,
        task=None,
        num_profiling_data_per_ratio=500,
        max_length_profiling_data=None,
        intra_request_cache_update_interval=32,
        threshold=0.95,
        cache_budget=5000,
        **kwargs,
    ):
        assert caching_policy
        assert caching_policy in ["bicache", "bicache_fastdllm"]
        assert task in ["arc", "gpqa", "math", "gsm8k"]

        super().__init__()
        
        self.device = device
        self.model = LLaDAModelLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.mask_id = mask_id

        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.is_instruct = True
        self.save_path = save_path
        self.show_speed = show_speed
        self.caching_policy = caching_policy
        self.task = task
        self.num_profiling_data_per_ratio = num_profiling_data_per_ratio
        
        system_prompt_config = {
            "arc": False,
            "gpqa": False,
            "math": False,
            "gsm8k": True,
        }

        system_prompt = get_system_prompt(task, fewshots=system_prompt_config[task])

        ids = np.load("./ratio_ordered_WildChat_ids.npy", allow_pickle=True)
        start_time = time.perf_counter()
        inter_request_caching_layer = (
            LLaDAProfiler(
                device,
                self.model,
                self.tokenizer,
            ).profile(
                dataset_name="allenai/WildChat-4.8M",
                ids=ids,
                max_sequence_length=max_length_profiling_data,
                num_profiling_data_per_ratio=num_profiling_data_per_ratio,
                threshold=threshold,
            )
        )
        self.profiling_time = time.perf_counter() - start_time
        
        if caching_policy == "bicache":
            self.model_engine = LLaDAEngine(
                device=self.device,
                model=self.model,
                tokenizer=self.tokenizer,
                number_of_inter_request_caching_layer=inter_request_caching_layer,
                intra_request_cache_update_interval=intra_request_cache_update_interval,
                cache_budget=cache_budget,
                show_speed=True
            )
        else:
            self.model_engine = FastdLLMLLaDAEngine(
                device=self.device,
                model=self.model,
                tokenizer=self.tokenizer,
                number_of_inter_request_caching_layer=inter_request_caching_layer,
                intra_request_cache_update_interval=intra_request_cache_update_interval,
                cache_budget=cache_budget,
                block_length=block_length,
                show_speed=True
            )

        self.model_engine.warm_up(1000, 100)
        
        self.prefix = [{"role": "system", "content": system_prompt}]
        
    
    def generate_until(self, requests):
        output = []
        ttft = []
        generation_time = 0.

        for req in tqdm(requests, desc= "Generating"):
            question = req.args[0]
            stop_tokens = req.args[1]['until']
            if self.task == "arc":
                question = question.replace("\\n", "\n")[97:-144] + "\nAnswer:"
            elif self.task == "gpqa":
                question = question[138:]
                
            m = self.prefix + [{"role": "user", "content": question}]

            start_time = time.perf_counter()
            generated_answer = self.model_engine.generate(
                sequence=m,
                steps=self.steps,
                gen_length=self.gen_length,
                mask_id=self.mask_id,
            )
            generation_time += time.perf_counter() - start_time

            generated_answer = self.tokenizer.decode(generated_answer[-self.gen_length:], skip_special_tokens=True)
            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]
            if self.task == "arc":
                match = re.search(r"The best answer is (.+)", generated_answer)
                generated_answer = match.group(1) if match else ""

            output.append(generated_answer)
            ttft.append(self.model_engine.ttft)

        if self.save_path is not None:
            with open(self.save_path, 'a', encoding='utf-8') as f:
                if self.show_speed:
                    num_requests = len(requests)
                    result = {
                        "Caching policy": self.caching_policy,
                        "Benchmark": self.task,
                        "Profiling time (minute)": round(self.profiling_time / 60, 2),
                        "Throughput (tokens/s)": round(num_requests * self.gen_length / generation_time, 2),
                    }
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

        return output
    
    def loglikelihood(self, requests):
        raise NotImplementedError
    
    def loglikelihood_rolling(self, requests):
        raise NotImplementedError


if __name__ == "__main__":
    cli_evaluate()
    
