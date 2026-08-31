# Third-Party Notices

This repository contains portions of third-party source code and other redistributed research artifacts. This notice lists only material copied, adapted, or derived into this repository. Packages installed separately from `requirements.txt` are not listed here.

## LLaDA

- Project: Large Language Diffusion Models (LLaDA)
- Sources:
  - https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct
  - https://github.com/ML-GSAI/LLaDA
- License: MIT
- Material used in this repository:
  - `model/llada/configuration_llada.py`
  - `model/llada/modeling_llada.py`
  - LLaDA sampling and evaluation logic adapted in `bicache/` and `eval.py`

The upstream LLaDA-8B-Instruct metadata identifies the model source as MIT licensed. The inspected upstream source files do not contain a separate copyright notice.

### MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## xAI Grok prompts

- Project: Grok prompts
- Source: https://github.com/xai-org/grok-prompts/blob/a7c186f5ccac95875c0041aed60398f6ecb6d6c7/grok4_system_turn_prompt_v8.j2
- License: GNU Affero General Public License v3.0 (`AGPL-3.0-only`)
- Modification date: 2026-03-14 (initial BiCache repository commit)
- Material used in this repository:
  - `system_prompts/templates/partials/safety_instructions.j2`

The upstream prompt was excerpted and adapted for BiCache. The local template contains selected, verbatim portions of that prompt. The full AGPL-3.0 license text is provided in the repository's `LICENSE` file.

## GSM8K

- Project: Training Verifiers to Solve Math Word Problems (GSM8K)
- Source: https://github.com/openai/grade-school-math
- Source revision inspected: `3101c7d5072418e28b9008a6636bde82a006892c`
- License: MIT
- Material used in this repository:
  - The first five training examples in
    `system_prompts/templates/tasks/gsm8k/examples.j2`

### MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## WildChat-4.8M

- Dataset: WildChat-4.8M
- Source: https://huggingface.co/datasets/allenai/WildChat-4.8M
- License: Open Data Commons Attribution License 1.0 (`ODC-By-1.0`)
- Material used in this repository:
  - `ratio_ordered_WildChat_ids.npy`, which stores row indices and turn counts
    used to select profiling records

The conversation records themselves are downloaded from the dataset at runtime and are not redistributed in this repository. The dataset revision is not pinned by the current code.
