# neural-from-scratch

A personal learning project. I am implementing and experimenting with a GPT-style large language model from scratch in PyTorch, on my own machine, so I can see how the model is built and how training actually behaves.

I am working through the pieces in order: text tokenization and data loading, attention, a small decoder-only transformer, pretraining on unlabeled text, and finetuning. The models stay small on purpose. I want runs I can finish locally, code I can read line by line, and room to change a layer or a hyperparameter and try again.

The core implementations in this repository are adapted from Sebastian Raschka's open-source project [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch). This repository is my derivative copy for study. I use that code as the starting point, run it, and modify it as I learn. Authorship of those original implementations stays with the upstream project. The copyright and license details are in [Attribution / License](#attribution--license).

## What I am working on

- A GPT-style model in PyTorch: embeddings, causal attention, transformer blocks, and text generation.
- A training loop for pretraining on unlabeled text, then finetuning for classification and instruction following.
- Optional experiments that already live in this tree, after the main path is clear: KV cache, grouped-query attention, mixture-of-experts, Llama-style and Qwen-style architectures, LoRA, and direct preference optimization.
- Keeping the setup local and small enough that I can inspect weights, loss, and generated text myself.

## Hardware

I run this on a local Ubuntu machine with an NVIDIA RTX 5070. PyTorch uses that GPU through CUDA when it is available. The main notebooks and training scripts target small models, which fits how I want to learn: short runs, readable code, and the option to rerun an experiment after a change.

## Layout

The folder names follow the upstream repository. Each numbered directory is one stage of the build. The main code for a stage is in its `01_main-chapter-code/` directory. Later directories also contain optional experiments.

| Path | What it covers |
| --- | --- |
| [`setup/`](setup) | Environment and install notes |
| [`ch02/`](ch02) | Tokenization, embeddings, and the dataloader |
| [`ch03/`](ch03) | Attention, including multi-head attention |
| [`ch04/`](ch04) | The GPT model, plus optional attention variants (KV cache, grouped-query attention, mixture-of-experts, and others) |
| [`ch05/`](ch05) | Pretraining, generation, and optional architectures (Llama, Qwen, Gemma, and others) |
| [`ch06/`](ch06) | Finetuning for text classification |
| [`ch07/`](ch07) | Instruction finetuning, dataset tools, and preference tuning |
| [`appendix-A/`](appendix-A) | PyTorch introduction used by the later code |
| [`appendix-D/`](appendix-D) | Training-loop additions such as learning-rate schedules |
| [`appendix-E/`](appendix-E) | Parameter-efficient finetuning with LoRA |
| [`pkg/llms_from_scratch/`](pkg/llms_from_scratch) | The core implementations as an importable package |

`ch01/` is background notes and has no model code. `appendix-B/` and `appendix-C/` are reference notes and exercise solutions carried over with the upstream tree.

Entry points I use:

- [ch02.ipynb](ch02/01_main-chapter-code/ch02.ipynb) for text data
- [ch03.ipynb](ch03/01_main-chapter-code/ch03.ipynb) for attention
- [ch04.ipynb](ch04/01_main-chapter-code/ch04.ipynb) and [gpt.py](ch04/01_main-chapter-code/gpt.py) for the model
- [ch05.ipynb](ch05/01_main-chapter-code/ch05.ipynb) and [gpt_train.py](ch05/01_main-chapter-code/gpt_train.py) for pretraining
- [ch06.ipynb](ch06/01_main-chapter-code/ch06.ipynb) for classification finetuning
- [ch07.ipynb](ch07/01_main-chapter-code/ch07.ipynb) for instruction finetuning

## Setup

```bash
git clone https://github.com/rudzy123/neural-from-scratch.git
cd neural-from-scratch
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers PyTorch, Jupyter, and the libraries used by the main notebooks. Some optional notebooks list extra packages in a local `requirements-extra.txt`. Further install notes are in [setup/README.md](setup/README.md).

## Attribution / License

This repository is a derivative educational project. It is adapted from the open-source repository [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka, licensed under the Apache License, Version 2.0.

The original code and materials are copyright Sebastian Raschka and are used under the Apache License 2.0:

```
Copyright 2023-2026 Sebastian Raschka

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

The full license text is [LICENSE.txt](LICENSE.txt). That file is the Apache License 2.0 text from the upstream project, including the original copyright notice. It is kept intact.

Modifications have been made, and further modifications will be made, for personal learning. Those changes are part of this derivative project. They do not transfer authorship of the upstream implementations, and they do not replace the Apache-2.0 terms that apply to the original code. Copyright and attribution notices shipped with the source, including the notice in `LICENSE.txt` and the notices in `pkg/llms_from_scratch/`, remain in place.
