# Bypassing Qwen behavior (activation steering)

This repository is a **Google Colab–style Jupyter notebook** experiment on **activation steering** for Qwen3 causal LMs. The goal is to probe how adding a learned “concept vector” to hidden states during generation shifts answers on benign vs. harmful prompts.

## What’s in the notebook

- **Environment setup**: PyTorch, Transformers, optional `transformer_lens`, `datasets`.
- **Model loading**: Qwen3 checkpoints (e.g. `Qwen/Qwen3-1.7B`) with chat templates and `enable_thinking=False` where applicable.
- **Concept vector**: Differences of mean residual-stream activations between **positive** (benign, yes/no) and **negative** (harmful, yes/no) calibration prompts, then applied via **forward hooks** at a chosen layer.
- **Steered generation**: Helpers such as `steered_generate_hf_final` (short answers) and `steered_generate_long` (longer samples).
- **Metrics**: Bar charts for “flip” rates (positive prompts steered toward “no”, negative prompts steered toward “yes”).
- **AdvBench sweep**: Loads `walledai/AdvBench`, generates steered completions, then scores them with **OpenAI** (`gpt-4o-mini`) as COMPLIANCE / REFUSAL / AMBIGUOUS (see the notebook for the exact rubric).
- **Optional multi-size run**: Loops over several Qwen3 sizes and plots compliance rate (requires GPU memory and time).

## Requirements

- **GPU** strongly recommended for Qwen inference.
- **Hugging Face** access for model weights (accept model terms as needed).
- **`OPENAI_API_KEY`**: For Colab, store this in **Secrets**; the notebook reads it with `google.colab.userdata`. For local Jupyter, set the env var and adjust the client initialization if you are not using Colab.

Install dependencies as you run the notebook cells (`pip install` lines are included in the notebook).

## Evaluation pipeline (AdvBench)

1. Run earlier cells until `steered_generate_long`, `model`, `tokenizer`, `concept_vector`, and `adv_bench_prompts` exist.
2. **Generation cell**: Builds `df_eval` with columns `prompt` and `response`.
3. **OpenAI cell**: Adds `gpt_label` to `df_eval` using `gpt-4o-mini`.

There is **no secondary local classifier** in this path; a single LLM judge is used for semantic labels.

## Ethics and safety

This work deals with **harmful requests** and model **refusal behavior** for research context. Use only in controlled environments, follow provider policies and applicable law, and do not use outputs to cause harm. The notebook includes harmful prompt lists **only** as standard benchmarks (e.g. AdvBench-style evaluations), not as instructions.

## Running locally

Open `main.ipynb` in Jupyter or VS Code. If you are not on Colab, replace `userdata.get('OPENAI_API_KEY')` with `os.environ["OPENAI_API_KEY"]` (or your preferred secret mechanism) in the OpenAI cells.

## File layout

| Path        | Description                                      |
|------------|---------------------------------------------------|
| `main.ipynb` | Full experiment notebook                       |
| `README.md`  | This file                                      |
