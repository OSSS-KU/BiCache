# Enabling KV Caching of Shared Prefix for Diffusion Language Models

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2606.07571)
![Conference](https://img.shields.io/badge/EMNLP-2026%20Main-blue)
![License](https://img.shields.io/badge/License-AGPL--3.0-green)

Official implementation of **BiCache**, accepted at **EMNLP 2026 Main Conference**.


## Implementations

We implement **BiCache**, a shared prefix caching method for diffusion language models (DLMs).

Following the paper, BiCache profiles the reusable shallow-layer depth `b` from the shared prefix ratio `r`. It reuses the shared prefix KV cache across requests in shallow layers and periodically refreshes shared prefix KV cache within a request in deep layers. It is also implemented to work together with **Fast-dLLM** as an orthogonal acceleration method.

- [`model/llada/modeling_llada.py`](model/llada/modeling_llada.py)  
  Extends the original `LLaDA` model so that **BiCache** can be plugged into the forward path.  
  The model interface is modified to accept `prefix_len`, `prefix_cache`, and `dual_cache`, which enables:
  1. Shared prefix KV reuse across requests in shallow layers.  
  2. Optional integration with block-wise `dual_cache` used by Fast-dLLM-style decoding.

- [`bicache/bicache_engines.py`](bicache/bicache_engines.py)  
  Implements **BiCache-based inference**. This engine:
  1. Tokenizes a chat sequence into shared prefix and user-prompt regions.  
  2. Computes the shared prefix ratio `r`.  
  3. Looks up the number of reusable shallow layers from the lookup table that maps `r` to `b`.  
  4. Manages the shared prefix KV cache with hash-based lookup and cache eviction.  
  5. Performs iterative denoising while periodically refreshing the shared prefix KV cache within a request for deep layers.

- [`bicache/bicache_fast_dllm_engines.py`](bicache/bicache_fast_dllm_engines.py)  
  Implements **BiCache + Fast-dLLM inference**.
  In addition to shared prefix KV caching from BiCache, this engine introduces block-wise decoding with `dual_cache` so that Fast-dLLM-style KV reuse can be combined with BiCache. This matches the paper’s orthogonal integration design, where BiCache further accelerates Fast-dLLM-based inference.

- [`bicache/bicache_profiler.py`](bicache/bicache_profiler.py)  
  Implements the **profiler** for determining the shallow-layer depth `b`.
  The profiler compares KV representations from prefix-only inputs and full inputs using cosine similarity, and determines `b`, which represents the number of shallow layers that can safely reuse shared prefix KVs under a similarity threshold. This corresponds to the paper’s offline profiling stage for constructing the lookup table that maps `r` to `b`.


## Setup
We recommend using **Python 3.10** to ensure compatibility with the dependencies used in this project.

### 1. Clone this repository
```
git clone https://github.com/OSSS-KU/BiCache.git
cd BiCache
```

### 2. Create a Python environment
Using `venv`:
```bash
python3.10 -m venv bicache-env
source bicache-env/bin/activate
```

Or using `conda`:
```
conda create -n bicache python=3.10
conda activate bicache
```

### 3. Install dependencies
Install the required Python packages using:
```
pip install -r requirements.txt
```

This will install all dependencies required to run the BiCache implementation and the inference engines.

The first run downloads LLaDA-8B-Instruct and WildChat-4.8M from
Hugging Face. Ensure that sufficient disk space and network access are
available.


## Benchmarks
We use the LM evaluation framework provided by
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to run benchmark evaluations.
Our evaluation scripts integrate BiCache-based inference engines with the lm-evaluation-harness pipeline.  

You can run the benchmark evaluation using the provided script:
```bash
bash run_eval.sh
```

This script executes inference benchmarks for the implemented engines (BiCache and BiCache + Fast-dLLM) and reports the evaluation results.

### Default Configuration

`run_eval.sh` uses the main BiCache hyperparameters reported in the paper.
You can modify the model path and engine options directly in the script.

Some parameters may need to be adjusted depending on the available GPU memory:

* **`max_length_profiling_data`**: Maximum sequence length allowed for samples used in the profiling stage. Profiling data longer than this length is filtered out.

* **`cache_budget`**: The maximum number of tokens that can be stored in the shared prefix KV cache.

These parameters are not specified in the paper because the appropriate values depend on the GPU memory capacity used during evaluation.
If GPU memory is limited, reduce these values to avoid out-of-memory errors.


## Citation

If you use BiCache, please cite our paper:

```bibtex
@misc{go2026enablingkvcachingshared,
      title={Enabling KV Caching of Shared Prefix for Diffusion Language Models}, 
      author={Younghun Go and Jaehoon Han and Changyong Shin and Chuck Yoo and Gyeongsik Yang},
      year={2026},
      eprint={2606.07571},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.07571}, 
}
```
