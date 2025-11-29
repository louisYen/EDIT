import torch
import argparse
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
import os
from sft_trainer import (
    dLLMTrainer,
    dLLMSFTDataset,
    dLLMDataCollator,
    preprocess_dataset,
)
import torch.distributed as dist
import random
import numpy as np
from datetime import datetime

from analysis.adamw_lora_logger_callback import (
    AdamWLoRAMonitorCallback
)

# Set W&B environment vars
os.environ["WANDB_PROJECT"] = f"EDIT"
os.environ["WANDB_NAME"] = "run_" + os.environ["WANDB_PROJECT"]

def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Check if XPU is available and set the seed for XPU
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)

    # Set deterministic behavior for cuDNN (if using CUDA)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True

def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])

    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        dist.init_process_group("ccl")
        torch.xpu.set_device(local_rank)
    else:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()

# Initialize argument parser
def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument(
        "--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct", help="Name of the pretrained model"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument(
        "--max_length", type=int, default=4096, help="Maximum sequence length for tokenization"
    )
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/llada",
        help="Directory to save model checkpoints and logs",
    )
    parser.add_argument("--job_name", type=str, default="llada-s1", help="Job Name")
    parser.add_argument("--train_data", type=str, default="simplescaling/s1K", help="Path to training data")
    parser.add_argument(
        "--debugging", action="store_true", help="Use while debugging model - only disables wandb logging"
    )
    parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs/XPUs to use")

    return parser.parse_args()

# Model loading with LoRA integration
def load_model_and_tokenizer(args, local_rank):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, padding_side="right", trust_remote_code=True, use_fast=True
    )

    # Load model
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(local_rank)

    # LoRA configuration
    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Applying LoRA model
    model = get_peft_model(model, lora_config)
    model = model.to(torch.bfloat16)  # Cast fp32 lora params to bf16

    model = model.to(local_rank)

    return tokenizer, model

# Dataset loading
def load_data(args, tokenizer):
    data = load_dataset(args.train_data, split="train")
    train_data, eval_data = preprocess_dataset(data, tokenizer, args.max_length)
    print("Train data length: ", len(train_data))
    print("Eval data length: ", len(eval_data))
    train_dataset = dLLMSFTDataset(train_data, tokenizer, args.max_length)
    eval_dataset = dLLMSFTDataset(eval_data, tokenizer, args.max_length, eval=True)
    return train_dataset, eval_dataset

# Training setup
def train_model(args, tokenizer, model):
    # Load dataset
    train_dataset, eval_dataset = load_data(args, tokenizer)

    """
    - transformers/training_args.py
    Line 1593-1598
        if self.evaluation_strategy is not None:
            warnings.warn(
                "`evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead",
                FutureWarning,
            )
            self.eval_strategy = self.evaluation_strategy
    """

    """
    training_args:
        - optim: OptimizerNames.ADAMW_TORCH (default)
            > Ref: https://github.com/huggingface/transformers/blob/5009252a05144f439e76502083c4380c33683054/src/transformers/training_args.py#L611-L614
    """

    # Training arguments setup
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.job_name),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        eval_strategy="no", # Disable evaluation during training
        logging_steps=2,
        save_steps=1, # Save checkpoints every step
        save_total_limit=None,
        load_best_model_at_end=False,
        learning_rate=args.learning_rate,
        weight_decay=0.1,
        max_grad_norm=1.0,
        optim="adamw_hf",
        bf16=True,
        report_to="wandb" if not args.debugging else "none",
        remove_unused_columns=False,
    )

    world_size = dist.get_world_size() if dist.is_initialized() else args.num_gpus
    num_train_steps = int(
        len(train_dataset)
        * args.num_epochs
        / (args.batch_size * args.grad_accum_steps * world_size)
    )

    # Initialize Trainer with custom dLLMTrainer
    trainer = dLLMTrainer(
        model=model,
        args=training_args,
        data_collator=dLLMDataCollator(
            tokenizer=tokenizer,
            mask_token_id=126336,
            max_length=args.max_length),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[AdamWLoRAMonitorCallback(
            logging_steps=1, # Change to 1
            output_dir=training_args.output_dir,
            save_steps=training_args.save_steps,
            debugging=args.debugging)],
    )

    # Start training
    trainer.train()

    cleanup_ddp()

if __name__ == "__main__":
    init_seed(42)

    local_rank = setup_ddp()

    # Parse command-line arguments
    args = parse_args()

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(args, local_rank)

    model_device = next(model.parameters()).device
    print(f'LLM is on device: {model_device}')

    # Train the model
    train_model(args, tokenizer, model)
