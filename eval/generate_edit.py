import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist

from rich.console import Console
from typing import Optional, Dict, List, Tuple

from collections import OrderedDict

from rich.progress import Progress

from predict_tokens import predict_tokens_per_step

console = Console()

def get_num_transfer_tokens(
    mask_index: torch.Tensor,
    steps: int,
    mode: str = "uniform",
    steepness: Optional[float] = None,
    gaussian_params: dict = {"center": None, "std": None},
) -> torch.Tensor:

    device = mask_index.device
    B = mask_index.size(0)
    total_mask = mask_index.sum(dim=1, keepdim=True)  # (B, 1)

    # Generate weights per step according to the selected mode
    step_ids = torch.arange(steps, device=device).float() / steps # ∈ [0, 1)

    if mode == "cosine_ramp":
        weights = torch.cos((1 - step_ids) * torch.pi / 2)  # More tokens in early steps
    elif mode == "decay":
        weights = torch.exp(-steepness * step_ids) # High -> Low
    elif mode == "sigmoid":
        weights = torch.sigmoid(2 * (step_ids - 0.5))  # More in middle
    elif mode == "gaussian":
        center, std = gaussian_params["center"], gaussian_params["std"]
        center = 1.0 / 2 if center is None else 1.0 / center
        std = 1.0 / 6  if std is None else 1.0 / std

        weights = torch.exp(-0.5 * ((step_ids - center) / std) ** 2)

    elif mode == "uniform":
        weights = torch.ones_like(step_ids)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Normalize weights so sum over steps = 1
    weights = weights / weights.sum()

    # Expand to (B, steps)
    weights = weights.unsqueeze(0).expand(B, -1)  # (B, steps)

    # Convert weights to token counts
    token_floats = total_mask * weights  # (B, steps)
    base = token_floats.floor().long()
    remainder = (token_floats - base).sum(dim=1)  # (B,)

    # Distribute remaining tokens
    token_counts = base.clone()
    for b in range(B):
        remaining = int(remainder[b].item())
        if remaining > 0:
            topk = torch.topk((token_floats[b] - base[b]), k=remaining).indices
            token_counts[b, topk] += 1

    return token_counts.to(torch.int64)

def clean_weight_suffix(optim_update_trace: Dict[str, torch.Tensor]) -> OrderedDict:
    """
    Remove the '.weight' suffix from keys in the optimizer update trace.

    Args:
        optim_update_trace (dict): key = layer name with '.weight' suffix

    Returns:
        OrderedDict: A new dictionary with '.weight' removed from keys.
    """
    cleaned_trace = OrderedDict()
    for key, value in optim_update_trace.items():
        # Remove '.weight' suffix from the key
        new_key = key.replace('.weight', '')
        cleaned_trace[new_key] = value

    return cleaned_trace

def register_lora_hooks(model, layer_outputs, lora_layers: list=["lora_A", "lora_B"]):
    # Registers forward hooks on selected LoRA layers to capture outputs
    def make_hook(name):
        def hook_fn(module, input, output):
            """
            - input: tuple
                - len(input) = 1
            - output: torch.Tensor
            """
            layer_outputs[name] = output.detach()#.clone().cpu()

        return hook_fn

    for name, module in model.named_modules():
        if any(x in name for x in lora_layers) and "default" in name:
            module.register_forward_hook(make_hook(name))

def compute_dynamic_temperature(mask_index: torch.Tensor, T_min: float = 0.1, T_max: float = 2.0) -> torch.Tensor:
    # Use the first sample, assuming all are identical during inference
    valid_tokens = mask_index[0].sum().float()
    fraction = valid_tokens / mask_index.shape[1]
    temperature = T_max - (T_max - T_min) * fraction
    return temperature.clamp(min=T_min, max=T_max).item()

def check_lora_b_reasoning_convergence(
    current_activations: dict,
    previous_activations: Optional[dict],
    optim_state: dict,
    mask_index: torch.Tensor,
    blk_start_idx: int,
    blk_end_idx: int,
    prompt_length: int,
    threshold: float = 0.05,
    temperature: float = 1. # Lower than 1 → sharper; Higher than 1 → smoother
) -> Tuple[bool, float]:

    kl_values = []

    B, S = mask_index.shape

    assert B == 1, "Support batch size == 1 only!"

    device = mask_index.device

    # Build response mask [B, R], where R = response_length

    # Create a block mask initialized to False (0)
    block_mask = torch.zeros_like(mask_index, dtype=torch.bool)

    # Set the specified range to True
    block_mask[:, blk_start_idx:blk_end_idx] = True

    # Combine the block mask with the negated mask_index
    block_mask = ~mask_index & block_mask

    # Slice the result to get visible_mask
    visible_mask = block_mask[:, prompt_length:] # True where tokens are visible (not masked)

    for name in current_activations:
        curr_act = current_activations[name]
        prev_act = previous_activations.get(name)
        if prev_act is None:
            continue

        # Slice only the response token region (excluding prompt)
        curr_act = curr_act[:, prompt_length:, :]
        prev_act = prev_act[:, prompt_length:, :]

        query = F.normalize(curr_act, dim=-1)
        key   = F.normalize(prev_act, dim=-1)

        # Stiff reasoning subspace (aggregated from AdamW updates)
        filter_update = optim_state[name].to(device)

        filters = F.normalize(filter_update, dim=0)

        # Logic direction projections of current activations
        q_logic = torch.matmul(query, filters)

        # Logic direction projections of previous activations
        k_logic = torch.matmul(key, filters)

        q_logic = q_logic * visible_mask.unsqueeze(-1) # zero out masked tokens
        k_logic = k_logic * visible_mask.unsqueeze(-1) # zero out masked tokens

        q_dist = F.softmax(q_logic / temperature, dim=-1) + 1e-8
        k_dist = F.softmax(k_logic / temperature, dim=-1) + 1e-8
        kl_div = F.kl_div(q_dist.log(), k_dist, reduction='none').sum(dim=-1)
        kl_div = kl_div * visible_mask

        kl_sum = kl_div.sum(dim=1)
        valid_tokens = visible_mask.sum(dim=1).clamp(min=1e-6)
        kl_mean_per_batch = kl_sum / valid_tokens

        kl_values.append(kl_mean_per_batch)

    if len(kl_values) == 0:
        return False, np.inf

    kl_values = torch.stack(kl_values).transpose(1, 0)
    kl_values = kl_values.mean(dim=1)

    kl_value_for_one_batch = kl_values.mean().item()

    return kl_value_for_one_batch  < threshold, kl_value_for_one_batch

def get_device_type():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        return "xpu"
    else:
        raise RuntimeError("No CUDA or XPU devices are available.")

@torch.no_grad()
def generate(
    model,
    prompt,
    tokenizer,
    steps=64,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    optim_state: dict[str, torch.Tensor] = None,  # AdamW update history
    early_termination_cfg: Optional[dict]={},
    target_lora_layers: Optional[dict] = None,  # LoRA layers to monitor
    verbose: bool=False,
    device_id: int=0,
):

    B, L = prompt.shape

    optim_state = clean_weight_suffix(optim_state)
    sample_exit_steps: List[int] = list()

    # Use mixed precision for faster computation
    device_type = get_device_type()
    with torch.autocast(device_type=device_type):

        # Create tensor x filled with mask tokens after prompt
        x = torch.full(
            (B, L + gen_length), # B x S, S = L + gen_length
            mask_id,
            dtype=torch.long,
            device=prompt.device
        ) # (B, L)

        # Copy the prompt tokens into the beginning of x
        x[:, : L] = prompt.clone()

        # Identify which positions are fixed (prompt) vs. masked
        prompt_index = x != mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length       # divide generation into equal blocks
        steps_per_block = max(1, steps // num_blocks) # how many steps per block

        if early_termination_cfg["num_blocks"]:
            num_blocks = early_termination_cfg["num_blocks"]
            first_block_length = early_termination_cfg["first_block_length"]

            num_blocks_for_first_block = gen_length // first_block_length
            first_block_steps = steps // num_blocks_for_first_block

            num_rest_blocks = num_blocks - (first_block_length // block_length)

            # Calculate initial steps per block
            remaining_steps = steps - first_block_steps

            # Create a list for steps per block
            steps_per_block = [first_block_steps] + \
                              [remaining_steps // num_rest_blocks] * num_rest_blocks
            block_lengths = [first_block_length] + \
                            [(gen_length - first_block_length) // num_rest_blocks] * num_rest_blocks

            num_blocks = len(block_lengths)
        else:
            # If there's only one block, all steps go into it
            steps_per_block = [steps_per_block] * num_blocks
            block_lengths = [block_length] * num_blocks

        # Compute start and end indices per block
        block_start_end = []
        curr = L  # Start after prompt
        for length in block_lengths:
            block_start_end.append((curr, curr + length))
            curr += length

        early_terminated_steps = torch.full(
            (B, num_blocks),
            1000,
            device=x.device, dtype=torch.long)

        # --- Initialize LoRA Monitoring ---
        curr_lora_activations = {}
        if optim_state is not None and early_termination_cfg is not None:
            register_lora_hooks(model, curr_lora_activations, lora_layers=["lora_B",])
            early_stop_enabled = True
        else:
            early_stop_enabled = False

        for block_idx in tqdm(
                range(num_blocks),
                desc=f"📦 [GPU:{device_id:2d}] Iterate Blocks",
                ncols=100,
                position=1):

            # start_idx = L + block_idx * block_length
            # end_idx = L + (block_idx + 1) * block_length
            start_idx, end_idx = block_start_end[block_idx]

            block_length = block_lengths[block_idx]

            # Identify which positions in this block are still masked
            block_mask_index = x[:, start_idx:end_idx] == mask_id

            # Precompute how many tokens to unmask at each step
            steps_for_current_block = steps_per_block[block_idx]

            token_schedule: dict = early_termination_cfg["token_schedule"][block_idx]

            center = token_schedule.get("center", 4)
            std = token_schedule.get("std", 5)

            stability_span = token_schedule.get("stability_span", 6)
            required_stable = token_schedule["required_stable"]
            polish_steps = token_schedule["polish_steps"]
            kl_thres = token_schedule["threshold"]

            num_transfer_tokens = get_num_transfer_tokens(
                block_mask_index,
                steps_for_current_block,
                mode="gaussian",
                gaussian_params={
                    "center": center,
                    "std": std,
                },
            )

            # --- Collect activations ---
            prev_lora_activations = {}
            kl_history = []

            softmax_temperature = compute_dynamic_temperature(
                mask_index=block_mask_index,
                T_min=0.1, T_max=2.0
            )

            for step_idx in tqdm(
                    range(steps_for_current_block),
                    desc=f"🚀 [GPU:{device_id:2d}] Iterate Steps",
                    ncols=100,
                    position=2):

                # Re-compute which tokens are still masked
                mask_index = x == mask_id # BxS

                x = predict_tokens_per_step(
                    model=model,
                    mask_index=mask_index,
                    x=x,
                    prompt_index=prompt_index,
                    mask_id=mask_id,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    remasking=remasking,
                    num_transfer_tokens=num_transfer_tokens,
                    step_idx=step_idx,
                    end_idx=end_idx,
                )

                # --- Early Exit Based on LoRA-B Reasoning Convergence ---
                if early_stop_enabled and curr_lora_activations is not None:
                    should_exit, kl_value = check_lora_b_reasoning_convergence(
                        current_activations=curr_lora_activations,
                        previous_activations=prev_lora_activations,
                        optim_state=optim_state,
                        mask_index=mask_index,
                        blk_start_idx=start_idx,
                        blk_end_idx=end_idx,
                        prompt_length=L,
                        threshold=kl_thres,
                        temperature=softmax_temperature,
                    )

                    kl_history.append(kl_value)

                    if len(kl_history) >= stability_span:
                        stable_count = sum(kl < kl_thres
                                            for kl in kl_history[-stability_span:]
                                                if kl != float('inf'))
                        if stable_count >= required_stable:
                            console.print(
                                f"\n\t🚧 [Early Exit Triggered @ Block {block_idx}, Step {step_idx}]"
                            )

                            block_mask_index_left = x[:, start_idx:end_idx] == mask_id

                            # Ensure assist steps do not exceed remaining steps in the block
                            remaining_steps = steps_for_current_block - step_idx - 1
                            polish_steps = min(polish_steps, remaining_steps)

                            num_transfer_tokens_left = get_num_transfer_tokens(
                                mask_index=block_mask_index_left,
                                steps=polish_steps,
                                mode="uniform"
                            )

                            for rest_step_idx in tqdm(
                                range(polish_steps),
                                desc="🌟 Diffuse Remaining Tokens",
                                ncols=100,
                                position=3,
                            ):

                                # Re-compute which tokens are still masked
                                mask_index = x == mask_id

                                x = predict_tokens_per_step(
                                    model=model,
                                    mask_index=mask_index,
                                    x=x,
                                    prompt_index=prompt_index,
                                    mask_id=mask_id,
                                    cfg_scale=cfg_scale,
                                    temperature=temperature,
                                    remasking=remasking,
                                    num_transfer_tokens=num_transfer_tokens_left,
                                    step_idx=rest_step_idx,
                                    end_idx=end_idx,
                                )

                            # Force final flush step
                            early_terminated_steps[:, block_idx] = torch.tensor(
                                    [step_idx + polish_steps + 1] * B) # +1: since step_idx starts at 0

                            break

                    prev_lora_activations = {k: v.clone() for k, v in curr_lora_activations.items()}

                    early_terminated_steps[:, block_idx] = torch.tensor(
                            [step_idx + 1] * B) # +1: since step_idx starts at 0

                # --- Clean memory ---
                curr_lora_activations.clear()
                torch.cuda.empty_cache()
            prev_lora_activations.clear()

        return x, early_terminated_steps

