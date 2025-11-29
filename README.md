<p align="center">
  <img src="assets/EDIT_banner.jpg" alt="EDIT banner" width="100%">
</p>

# EDIT: Early Diffusion Inference Termination for dLLMs Based on Dynamics of Training Gradients

<p align="center">
<a href="https://arxiv.org/abs/"><img src="https://img.shields.io/static/v1?label=Paper&message=Link&color=green" height=20.5></a>
</p>

###### tags: `diffusion language models`, `early termination`, `training metadata`, `reasoning benchmarks`

This repo is the official implementation of "**EDIT: Early Diffusion Inference Termination for dLLMs Based on Dynamics of Training Gradients.**"

**TL;DR**: EDIT uses training-time metadata to enable early termination during inference in diffusion language models, reducing cost while maintaining or improving accuracy.

---

## 🛠️ <a name="1"></a> 1. Setup

The project is tested on Python 3.10 and supports two hardware platforms:

- NVIDIA CUDA GPUs
- Intel XPU

> All installation commands should be run from the project’s root directory: `EDIT/`

### <a name="1.1"></a> 1.1 Create Conda Environment

```bash=
conda create -n EDIT python=3.10 -y
conda activate EDIT
```

### <a name="1.2"></a> 1.2 Installation (Choose One Platform)

#### <a name="1.2.A"></a> 1.2.A | NVIDIA GPU Installation
- Command Line for Installing PyTorch 2.6 and Dependencies
```bash=
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

#### (Optional) Verify Installation
```bash=
python setup_check.py


# Example output:

🔍 Checking Python environment...

✅ torch: 2.6.0+cu124
...
🖥️ Device Detection:
✅ CUDA GPU detected: NVIDIA A100-SXM4-80GB (CUDA 12.4)
```

#### <a name="1.2.B"></a> 1.2.B | Intel XPU Installation
- Command Line for Installing PyTorch 2.6 and Dependencies
```bash=
python -m pip install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/xpu

python -m pip install \
    --proxy http://proxy-dmz.intel.com:912 \
    intel-extension-for-pytorch==2.6.10+xpu \
    oneccl_bind_pt==2.6.0+xpu \
    --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

pip install -r requirements.txt
```

#### (Optional) Verify Installation
```bash=
python setup_check.py


# Example output:

🔍 Checking Python environment...

✅ torch: 2.6.0+xpu

🖥️ Device Detection:
✅ Intel XPU detected
   XPU devices available: 12
```

### :world_map: <a name="1.3"></a> 1.3 Environment Summary
<details>
<summary>Library versions</summary>

- Framework
    - pytorch: 2.6.0 (CUDA 12.4 or XPU backend)
    - torchvision: 0.21.1
    - python: 3.10
- Hardware
    - GPU: NVIDIA A100-SXM4-80GB x 2
    - or Intel XPU x 12
</details>

---

<h2 align="center">The examples below use 2 GPUs</h2>

<p align="center">
  <em>For SFT, please specify your number of GPUs with <code>--nproc_per_node</code></em><br>
  <em>For inference, please specify your available GPUs using <code>--device_ids</code></em>
</p>


## 🔥 <a name="2"></a> 2. Supervised Fine-tuning (SFT)

To run SFT on LLaDA, use the following command.

> Reminder: Please ensure you are in the project root directory (`EDIT/`) and that your environment is active (`conda activate EDIT`) before running the commands below.

```bash=
cd SFT
torchrun \
  --nproc_per_node 2 \
  --master_port 29410 \
  train_llada_sft.py \
  --model_name "GSAI-ML/LLaDA-8B-Instruct" \
  --output_dir logs/logs_llada_sft \
  --num_epochs 2 \
  --debugging
```

**Argument notes**
- `--nproc_per_node` : number of processes per node (e.g., 2 for two GPUs)
- `--debugging`: optional flag to disable wandb logging

## 🚀 <a name="3"></a> 3. Run Early Termination on Fine-tuned Model

Before running early termination to shorten inference steps, switch to the `eval` directory:

```bash=
cd ../eval
```

### (Optional) For NVIDIA GPU users, please run:
```bash=
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
```

### Inference on Countdown Dataset

<details>
<summary>Countdown (Sequence 128)</summary>

```bash=
python dllm_eval_mp.py \
  --model_path "GSAI-ML/LLaDA-8B-Instruct" \
  --checkpoint_path ../SFT/logs/logs_llada_sft/llada-s1/checkpoint-200 \
  --dataset countdown \
  --output_path EDIT_results/countdown/seq128 \
  --config_file configs/EDIT_countdown_seq128.json \
  --gen_length 128 \
  --device_ids 0 1
```
</details>

<details>
<summary>Countdown (Sequence 256)</summary>

```bash=
python dllm_eval_mp.py \
  --model_path "GSAI-ML/LLaDA-8B-Instruct" \
  --checkpoint_path ../SFT/logs/logs_llada_sft/llada-s1/checkpoint-200 \
  --dataset countdown \
  --output_path EDIT_results/countdown/seq256 \
  --config_file configs/EDIT_countdown_seq256.json \
  --gen_length 256 \
  --device_ids 0 1
```
</details>

<details>
<summary>Countdown (Sequence 512)</summary>

```bash=
python dllm_eval_mp.py \
  --model_path "GSAI-ML/LLaDA-8B-Instruct" \
  --checkpoint_path ../SFT/logs/logs_llada_sft/llada-s1/checkpoint-200 \
  --dataset countdown \
  --output_path EDIT_results/countdown/seq512 \
  --config_file configs/EDIT_countdown_seq512.json \
  --gen_length 512 \
  --device_ids 0 1
```
</details>

### Inference on Sudoku Dataset

<details>
<summary>Sudoku (Sequence 128)</summary>

```bash=
python dllm_eval_mp.py \
  --model_path "GSAI-ML/LLaDA-8B-Instruct" \
  --checkpoint_path ../SFT/logs/logs_llada_sft/llada-s1/checkpoint-200 \
  --dataset sudoku \
  --output_path EDIT_results/sudoku/seq128 \
  --config_file configs/EDIT_sudoku_seq128.json \
  --gen_length 128 \
  --device_ids 0 1
```
</details>

<details>
<summary>Sudoku (Sequence 256)</summary>

```bash=
python dllm_eval_mp.py \
  --model_path "GSAI-ML/LLaDA-8B-Instruct" \
  --checkpoint_path ../SFT/logs/logs_llada_sft/llada-s1/checkpoint-200 \
  --dataset sudoku \
  --output_path EDIT_results/sudoku/seq256 \
  --config_file configs/EDIT_sudoku_seq256.json \
  --gen_length 256 \
  --device_ids 0 1
```
</details>

<details>
<summary>Sudoku (Sequence 512)</summary>

```bash=
python dllm_eval_mp.py \
  --model_path "GSAI-ML/LLaDA-8B-Instruct" \
  --checkpoint_path ../SFT/logs/logs_llada_sft/llada-s1/checkpoint-200 \
  --dataset sudoku \
  --output_path EDIT_results/sudoku/seq512 \
  --config_file configs/EDIT_sudoku_seq512.json \
  --gen_length 512 \
  --device_ids 0 1
```
</details>

### Inference on MATH500 Dataset

<details>
<summary>MATH500 (Sequence 128)</summary>

```bash=
python dllm_eval_mp.py \
  --model_path "GSAI-ML/LLaDA-8B-Instruct" \
  --checkpoint_path ../SFT/logs/logs_llada_sft/llada-s1/checkpoint-200 \
  --dataset math \
  --output_path EDIT_results/math/seq128 \
  --config_file configs/EDIT_math_seq128.json \
  --gen_length 128 \
  --device_ids 0 1
```
</details>

<details>
<summary>MATH500 (Sequence 256)</summary>

```bash=
python dllm_eval_mp.py \
  --model_path "GSAI-ML/LLaDA-8B-Instruct" \
  --checkpoint_path ../SFT/logs/logs_llada_sft/llada-s1/checkpoint-200 \
  --dataset math \
  --output_path EDIT_results/math/seq256 \
  --config_file configs/EDIT_math_seq256.json \
  --gen_length 256 \
  --device_ids 0 1
```
</details>

<details>
<summary>MATH500 (Sequence 512)</summary>

```bash=
python dllm_eval_mp.py \
  --model_path "GSAI-ML/LLaDA-8B-Instruct" \
  --checkpoint_path ../SFT/logs/logs_llada_sft/llada-s1/checkpoint-200 \
  --dataset math \
  --output_path EDIT_results/math/seq512 \
  --config_file configs/EDIT_math_seq512.json \
  --gen_length 512 \
  --device_ids 0 1
```
</details>

### Inference on GSM8K Dataset

<details>
<summary>GSM8K (Sequence 128)</summary>

```bash=
python dllm_eval_mp.py \
  --model_path "GSAI-ML/LLaDA-8B-Instruct" \
  --checkpoint_path ../SFT/logs/logs_llada_sft/llada-s1/checkpoint-200 \
  --dataset gsm8k \
  --output_path EDIT_results/gsm8k/seq128 \
  --config_file configs/EDIT_gsm8k_seq128.json \
  --gen_length 128 \
  --device_ids 0 1
```
</details>

<details>
<summary>GSM8K (Sequence 256)</summary>

```bash=
python dllm_eval_mp.py \
  --model_path "GSAI-ML/LLaDA-8B-Instruct" \
  --checkpoint_path ../SFT/logs/logs_llada_sft/llada-s1/checkpoint-200 \
  --dataset gsm8k \
  --output_path EDIT_results/gsm8k/seq256 \
  --config_file configs/EDIT_gsm8k_seq256.json \
  --gen_length 256 \
  --device_ids 0 1
```
</details>

<details>
<summary>GSM8K (Sequence 512)</summary>

```bash=
python dllm_eval_mp.py \
  --model_path "GSAI-ML/LLaDA-8B-Instruct" \
  --checkpoint_path ../SFT/logs/logs_llada_sft/llada-s1/checkpoint-200 \
  --dataset gsm8k \
  --output_path EDIT_results/gsm8k/seq512 \
  --config_file configs/EDIT_gsm8k_seq512.json \
  --gen_length 512 \
  --device_ids 0 1
```
</details>

## 📊 <a name="4"></a> 4. Perform Evaluation

### Evaluate the Countdown Dataset

<details>
<summary>Countdown (Sequence 128)</summary>

```bash=
python parse_and_get_acc.py -d EDIT_results/countdown/seq128/
```
</details>

<details>
<summary>Countdown (Sequence 256)</summary>

```bash=
python parse_and_get_acc.py -d EDIT_results/countdown/seq256/
```
</details>

<details>
<summary>Countdown (Sequence 512)</summary>

```bash=
python parse_and_get_acc.py -d EDIT_results/countdown/seq512/
```
</details>

### Evaluate the Sudoku Dataset

<details>
<summary>Sudoku (Sequence 128)</summary>

```bash=
python parse_and_get_acc.py -d EDIT_results/sudoku/seq128/
```
</details>

<details>
<summary>Sudoku (Sequence 256)</summary>

```bash=
python parse_and_get_acc.py -d EDIT_results/sudoku/seq256/
```
</details>

<details>
<summary>Sudoku (Sequence 512)</summary>

```bash=
python parse_and_get_acc.py -d EDIT_results/sudoku/seq512/
```
</details>

### Evaluate the MATH500 Dataset

<details>
<summary>MATH500 (Sequence 128)</summary>

```bash=
python parse_and_get_acc.py -d EDIT_results/math/seq128/
```
</details>

<details>
<summary>MATH500 (Sequence 256)</summary>

```bash=
python parse_and_get_acc.py -d EDIT_results/math/seq256/
```
</details>

<details>
<summary>MATH500 (Sequence 512)</summary>

```bash=
python parse_and_get_acc.py -d EDIT_results/math/seq512/
```
</details>

### Evaluate the GSM8K Dataset

<details>
<summary>GSM8K (Sequence 128)</summary>

```bash=
python parse_and_get_acc.py -d EDIT_results/gsm8k/seq128/
```
</details>

<details>
<summary>GSM8K (Sequence 256)</summary>

```bash=
python parse_and_get_acc.py -d EDIT_results/gsm8k/seq256/
```
</details>

<details>
<summary>GSM8K (Sequence 512)</summary>

```bash=
python parse_and_get_acc.py -d EDIT_results/gsm8k/seq512/
```
</details>

## 🔢 <a name="5"></a> 5. Compute Reduced Denoising Steps

### Calculate on Countdown Dataset

<details>
<summary>Countdown (Sequence 128)</summary>

```bash=
python compute_early_exit_diffusion_steps.py -d EDIT_results/countdown/seq128/
```
</details>

<details>
<summary>Countdown (Sequence 256)</summary>

```bash=
python compute_early_exit_diffusion_steps.py -d EDIT_results/countdown/seq256/
```
</details>

<details>
<summary>Countdown (Sequence 512)</summary>

```bash=
python compute_early_exit_diffusion_steps.py -d EDIT_results/countdown/seq512/
```
</details>

### Calculate on Sudoku Dataset

<details>
<summary>Sudoku (Sequence 128)</summary>

```bash=
python compute_early_exit_diffusion_steps.py -d EDIT_results/sudoku/seq128/
```
</details>

<details>
<summary>Sudoku (Sequence 256)</summary>

```bash=
python compute_early_exit_diffusion_steps.py -d EDIT_results/sudoku/seq256/
```
</details>

<details>
<summary>Sudoku (Sequence 512)</summary>

```bash=
python compute_early_exit_diffusion_steps.py -d EDIT_results/sudoku/seq512/
```
</details>

### Calculate on MATH500 Dataset

<details>
<summary>MATH500 (Sequence 128)</summary>

```bash=
python compute_early_exit_diffusion_steps.py -d EDIT_results/math/seq128/
```
</details>

<details>
<summary>MATH500 (Sequence 256)</summary>

```bash=
python compute_early_exit_diffusion_steps.py -d EDIT_results/math/seq256/
```
</details>

<details>
<summary>MATH500 (Sequence 512)</summary>

```bash=
python compute_early_exit_diffusion_steps.py -d EDIT_results/math/seq512/
```
</details>

### Calculate on GSM8K Dataset

<details>
<summary>GSM8K (Sequence 128)</summary>

```bash=
python compute_early_exit_diffusion_steps.py -d EDIT_results/gsm8k/seq128/
```
</details>

<details>
<summary>GSM8K (Sequence 256)</summary>

```bash=
python compute_early_exit_diffusion_steps.py -d EDIT_results/gsm8k/seq256/
```
</details>

<details>
<summary>GSM8K (Sequence 512)</summary>

```bash=
python compute_early_exit_diffusion_steps.py -d EDIT_results/gsm8k/seq512/
```
</details>


## 🙌 <a name="6"></a> 6. Acknowledgements

Big thanks to the authors of [LLaDA](https://github.com/ML-GSAI/LLaDA) and [d1](https://github.com/dllm-reasoning/d1/tree/e20637b0cc4257bdcd00d49d3b571e55776d054f) for their great work. Really appreciate it!
