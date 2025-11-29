import argparse
import json
import math
import os
import random
import time
from datetime import timedelta

import re
import numpy as np
import pandas as pd
import os.path as osp
import torch
import torch.distributed as dist
from torch.utils.data import (
    DataLoader, Subset,
    DistributedSampler, Sampler
)

from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModel,
    TrainingArguments,
    PreTrainedTokenizerBase
)
from peft import (
    PeftModel,
    get_peft_model_state_dict
)
from natsort import natsorted
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any

from generate_edit import generate

from gsm8k import GSM8KDataset
from math500 import MATH500Dataset
from countdown import CTDDataset
from sudoku import SudokuDataset

from typing import Optional, Dict

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.rule import Rule
from rich import print

import torch.multiprocessing as mp

from utils import load_or_collect_optim_states

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "countdown": CTDDataset,
    "sudoku": SudokuDataset,
}

console = Console()

def init_seed(seed):
    torch.use_deterministic_algorithms(True)

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Check if XPU is available and set the seed for XPU
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.manual_seed(seed)
        torch.xpu.manual_seed_all(seed)

def get_device_ids(device_ids: Optional[List[int]]=None):

    if device_ids is None:
        if torch.cuda.is_available():
            n_devices = torch.cuda.device_count()
            device_ids = list(range(n_devices))
            print(f"[INFO] No device_ids passed. Using all available CUDA devices: {device_ids}")
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            n_devices = torch.xpu.device_count()
            device_ids = list(range(n_devices))
            print(f"[INFO] No device_ids passed. Using all available XPU devices: {device_ids}")
        else:
            raise RuntimeError("No CUDA or XPU devices are available.")
    else:
        device_ids = device_ids
        print(f"[INFO] Using specified GPUs: {device_ids}")

    return device_ids

def get_lora_named_params(model):
    """
    Match LoRA parameters in the model (with `.default`) to the keys
    in PEFT LoRA state dict (which drops `.default`)
    """

    # Get LoRA keys like '...lora_A.weight' (no '.default')
    lora_keys = set(get_peft_model_state_dict(model).keys())

    matched_params = []
    for name, param in model.named_parameters():
        # Strip `.default` to normalize
        normalized_name = name.replace(".default", "")
        if normalized_name in lora_keys:
            matched_params.append((name, param))

    return matched_params

def extract_step_from_path(path):
    match = re.search(r"checkpoint-(\d+)", path)
    if match:
        return int(match.group(1))
    return None

def print_example(idx, questions, generated_texts, gt_answers):

    console.rule(f"[bold cyan] Example #{idx} [/bold cyan]", style="cyan")

    print(Rule("[bold yellow]🔍 Question[/bold yellow]"))
    print(Markdown(f"> {questions[idx]}"))

    # Generated Answer
    print(Panel(
        generated_texts[idx],
        title="🧠 [bold green]Generated Answer[/bold green]",
        border_style="green"
    ))

    # Ground Truth
    print(Rule("[bold indian_red]📘 Ground Truth[/bold indian_red]", style="indian_red"))
    print(gt_answers[idx])

    console.rule(style="dim")
    print("\n\n")

def evaluate(
    model,
    tokenizer,
    dataloader,
    gen_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    steps=64,
    block_length=32,
    optim_state: Optional[dict]=None,
    early_step_path_with_filename_prefix: Optional[str]=None,
    device_id: int=0,
    early_termination_cfg: Optional[dict]={},
):
    model.eval()

    total_processed = torch.tensor(0, device=model.device)
    wall_times = []
    all_generations = []
    device = model.device

    local_terminated_steps = [] # per-process collection

    for batch in tqdm(dataloader,
                      # disable=(dist.get_rank() != 0),
                      desc=f"📟 [GPU:{device_id:2d}] Iterate dataloader",
                      total=len(dataloader.dataset),
                      ncols=150,
                      position=0):

        start_time = time.time()
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        out, early_terminated_steps = generate(
            model,
            input_ids,
            tokenizer,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking="low_confidence",
            # +++ Add early termination +++
            optim_state=optim_state,
            device_id=device_id,
            early_termination_cfg=early_termination_cfg,
        )

        local_terminated_steps.append(early_terminated_steps.cpu()) # Detach from GPU

        generated_texts = tokenizer.batch_decode(out[:, -gen_length:], skip_special_tokens=False)
        example_result = [
            {
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": generated_texts[j],
                "ground_truth": gt_answers[j],
            }
            for j in range(len(gt_answers))
        ]
        all_generations.extend(example_result)
        total_processed += len(generated_texts)
        wall_times.append(time.time() - start_time)

        # Print individual results
        idx = random.randint(0, len(questions) - 1)
        print_example(
            idx=idx,
            questions=questions,
            generated_texts=generated_texts,
            gt_answers=gt_answers
        )

    # +++ (Early Termination) Save terminated steps +++
    all_steps = torch.cat(local_terminated_steps, dim=0)  # shape: [total_samples, num_blocks]
    total_samples = all_steps.shape[0]

    sample_avg = all_steps.float().mean(dim=1)  # average exit step per sample
    global_avg = sample_avg.mean().item()       # global average across all samples

    early_exit_text = Text()
    early_exit_text.append("[Early Termination]", style="dodger_blue1")
    early_exit_text.append(f"\nTotal number of gathered samples: {total_samples}")
    early_exit_text.append(f"\nGlobal average early terminated steps per sample: {global_avg:.2f}\n")
    early_exit_panel = Panel(early_exit_text, title="Early Termination Info", width=100)
    print(early_exit_panel)

    # all_steps: [total_samples, num_blocks] (torch.Tensor)
    sample_avg = all_steps.float().mean(dim=1, keepdim=True)  # [N, 1]
    steps_with_avg = torch.cat([all_steps, sample_avg], dim=1)  # [N, B+1]

    # Add global average as final row
    global_avg_row = torch.cat([
        all_steps.float().mean(dim=0, keepdim=True), # [1, B]
        sample_avg.mean(dim=0, keepdim=True)         # [1, 1]
    ], dim=1)  # [1, B+1]

    steps_with_avg = torch.cat([steps_with_avg, global_avg_row], dim=0)  # [N+1, B+1]

    # Convert to pandas for CSV saving
    num_blocks = all_steps.shape[1]
    columns = [f"Block_{i+1}" for i in range(num_blocks)] + ["Sample_Avg"]
    rows = [f"Sample_{i+1}" for i in range(all_steps.shape[0])] + ["Global_Avg"]

    df = pd.DataFrame(steps_with_avg.cpu().numpy(), index=rows, columns=columns)
    out_name = f"{early_step_path_with_filename_prefix}_early_terminated_steps_device-{device_id}.csv"
    df.to_csv(out_name)
    print(f"[Saved] {out_name}")

    avg_wall_time = sum(wall_times) / len(wall_times)
    metrics = {
        "wall_time": avg_wall_time,
        "generations": all_generations,
        "total_processed": total_processed.item(),
    }

    return metrics

class CustomSampler(Sampler):
    def __init__(
        self,
        dataset,
        shuffle=True,
        seed=0,
        drop_last=False,
    ) -> None:

        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Calculate the number of samples
        self.num_samples = len(self.dataset)
        self.total_size = len(self.dataset)

    def __iter__(self):
        # Generate indices for sampling
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Ensure indices are evenly divisible
        if not self.drop_last:
            indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        return iter(indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

def get_step_from_ckpt(ckpt_dir):
    match = re.search(r"checkpoint-(\d+)", ckpt_dir)
    if match:
        return int(match.group(1))
    else:
        return None  # or raise an error if critical

def inference_worker(
    device_id: int,
    dataset_subset: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    optim_state: Dict[str, List[torch.Tensor]],
    args: Any,
) -> None:

    # --- Fix random seed ---
    init_seed(42)

    # --- Determine Computing Device ---
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.set_device(device_id)

    # --- Load base model ---
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16).to(device_id)

    # --- Load PEFT model (i.e., LoRA) ---
    if args.checkpoint_path:
        model = PeftModel.from_pretrained(
            model,
            args.checkpoint_path,
            torch_dtype=torch.bfloat16).to(device_id)
        print(f"✨ [bold green][INFO][/bold green] Adapter loaded successfully")

    message = f"[bold magenta]🚀 Running model on device:[/bold magenta] [bold cyan]{device_id}[/bold cyan]"
    print(message)

    # --- Create Dataloader ---
    dataloader = DataLoader(
        dataset_subset,
        batch_size=args.batch_size,
        sampler=CustomSampler(dataset_subset, shuffle=False),
        collate_fn=dataset_subset.dataset.collate_fn
    )

    # --- Construct the model name based on checkpoint path and other parameters ---
    if args.checkpoint_path:
        model_name = args.checkpoint_path.split("/")[-2] + "_" + args.checkpoint_path.split("/")[-1]
    else:
        model_name = "base"

    # --- Append few-shot information if applicable (Default: 0-shot) ---
    if args.few_shot > 0:
        model_name += f"_fs{args.few_shot}"

    # --- Append suffix if provided ---
    if len(args.suffix) > 0:
        model_name += f"_{args.suffix}"

    # --- Construct the filename for saving generations ---
    filename = osp.join(
        args.output_path,
        f"{args.dataset}_"
        f"{model_name}_"
        f"{args.gen_length}_"
        f"{args.diffusion_steps}_"
        f"{device_id}_"
        "generations.json"
    )

    model_name = args.checkpoint_path.split("/")[-2] + "_" + args.checkpoint_path.split("/")[-1] if args.checkpoint_path else "base"
    if args.few_shot > 0:
        model_name += f"_fs{args.few_shot}"
    if len(args.suffix) > 0:
        model_name += f"_{args.suffix}"

    filename = osp.join(
        f"{args.output_path}",
        f"{args.dataset}_{model_name}_{args.gen_length}_{args.diffusion_steps}_{device_id}_generations.json"
    )
    print(f"Saving generations to {filename}")

    # +++ (Early Termination) Load EDIT config
    with open(args.config_file, "r") as fp:
        early_termination_cfg = json.load(fp)

    # --- Start perform evaluation ---
    metrics = evaluate(
        model,
        tokenizer,
        dataloader,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.diffusion_steps,
        optim_state=optim_state,
        early_step_path_with_filename_prefix=osp.join(
            args.output_path, f"{args.dataset}_{model_name}"
        ),
        device_id=device_id,
        early_termination_cfg=early_termination_cfg,
    )

    if not args.dont_save:
        with open(filename, "w") as f:
            json.dump(
                {
                    "generations": metrics["generations"],
                    "metrics": {
                        "wall_time": metrics["wall_time"],
                        "total_processed": metrics["total_processed"],
                    },
                    "model_path": args.model_path,
                    "checkpoint_path": args.checkpoint_path,
                    "gen_length": args.gen_length,
                    "diffusion_steps": args.diffusion_steps,
                    "block_length": args.block_length,
                },
                f,
                indent=2,
            )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--few_shot",
        type=int,
        default=0)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="For EDIT, set batch_size = 1")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "math", "countdown", "sudoku",],
        default="gsm8k"
    )

    # --- Args for DLLM Settings ---
    parser.add_argument(
        "--model_path",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="")
    parser.add_argument(
        "--gen_length",
        type=int,
        default=128)
    parser.add_argument(
        "--block_length",
        type=int,
        default=64)

    # --- Misc ---
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--add_reasoning", action="store_true")
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--output_path", type=str, default="results/")
    parser.add_argument("--dont_use_box", action="store_true")

    # --- Args for Using Multiprocessing (Spawn) ---
    parser.add_argument(
        "--device_ids",
        nargs="+",
        type=int,
        default=None,
        help="List of GPU device IDs to use (default: all available GPUs)"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/EDIT_countdown_seq128.json")

    args = parser.parse_args()

    # --- Initialization ---
    args.diffusion_steps = args.gen_length // 2 # (Keep this constant) Set this diffusion_steps as a constant
    device_ids = get_device_ids(args.device_ids)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # +++ Load Adam/AdamW trajectories +++
    optim_state = load_or_collect_optim_states(
        checkpoint_path=args.checkpoint_path,
    )
    print("✅ Optimizer state loaded successfully.")

    # --- Load evaluation dataset (target task) ---
    num_evals = {"gsm8k": -1, "math": -1, "countdown": 256, "sudoku": 256}
    dataset = DATASET_MAP[args.dataset](
        tokenizer,
        subsample=num_evals[args.dataset],
        num_examples=args.few_shot,
        add_reasoning=True,  # prefill for all models
    )

    # --- Get the number of GPUs avaliable ---
    num_gpus = len(device_ids)

    # --- Split dataset into chunks for each GPU ---
    dataset_chunks = torch.chunk(torch.tensor(range(len(dataset))), num_gpus)

    # --- Verify total samples ---
    total_samples = sum(len(chunk) for chunk in dataset_chunks)
    assert total_samples == len(dataset), "Total samples across chunks do not match the original dataset size."

    # --- (Split Dataset for Different Devices) Create dataset subsets ---
    dataset_subsets = [Subset(dataset, chunk.tolist()) for chunk in dataset_chunks]

    # --- (Multiprocessing) Set up multiprocessing using the 'spawn' method for safety and compatibility ---
    ctx = mp.get_context("spawn")

    # --- (Multiprocessing) List to keep track of all processes ---
    processes = []

    # --- Create output directory ---
    Path(args.output_path).mkdir(exist_ok=True, parents=True)

    # --- (Multiprocessing) Create a process for each GPU ---
    for gpu_id in device_ids:
        t_start = time.time()
        # --- Start a new process to run inference on a specific GPU ---
        p = ctx.Process(
            target=inference_worker,
            args=(gpu_id,
                  dataset_subsets[gpu_id],
                  tokenizer,
                  optim_state,
                  args)
        )
        p.start()  # Begin the process
        processes.append((p, t_start, gpu_id)) # Add the process to our list

    # --- Wait for all processes to finish ---
    for p, t_start, gpu_id in processes:
        p.join()  # Block until the process completes
        elapsed = time.time() - t_start
        print(f"⏳ GPU {gpu_id} finished in {timedelta(seconds=elapsed)}")

