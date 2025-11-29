import torch

import numpy as np
import torch.nn.functional as F

from rich.progress import Progress
from typing import Optional, Dict, List, Tuple

def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    # Stochastic sampler (noise injection)

    if temperature == 0.0:
        return logits  # Skip noise when temperature is 0

    # Use float32 instead of float64 for better performance
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32) # Uniform[0,1) noise
    gumbel_noise = (-torch.log(noise)) ** temperature # Apply Gumbel distribution with temperature
    return logits.exp() / gumbel_noise # Reparameterized logits for sampling

def predict_tokens_per_step(
    model: torch.nn.Module,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    prompt_index: torch.Tensor,
    mask_id: int,
    cfg_scale: float,
    temperature: float,
    remasking: str,
    num_transfer_tokens: torch.Tensor,
    step_idx: int,
    end_idx: int
) -> torch.Tensor:

    B, S = x.shape

    # Handle classifier-free guidance more efficiently
    if cfg_scale > 0.0:
        un_x = x.clone()
        un_x[prompt_index] = mask_id # mask prompt for unconditional input
        x_ = torch.cat([x, un_x], dim=0) # [conditional, unconditional]

        # Get logits in a single forward pass
        logits = model(x_).logits
        logits, un_logits = torch.chunk(logits, 2, dim=0)

        # Combine with classifier-free guidance
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
    else:
        logits = model(x).logits # BxSxV, V=126464

    # Add Gumbel noise and sample predicted tokens
    logits_with_noise = add_gumbel_noise(logits, temperature) # BxSxV
    x0 = torch.argmax(logits_with_noise, dim=-1) # BxS

    # Handle remasking strategy
    if remasking == "low_confidence":
        # Use float32 instead of float64 for better performance
        p = F.softmax(logits, dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device)
    else:
        raise NotImplementedError(remasking)

    # Ensure we don't process tokens beyond the current block
    x0_p[:, end_idx:] = -np.inf

    # Replace only masked positions with predictions
    x0 = torch.where(mask_index, x0, x)

    # Use confidence only for masked tokens
    confidence = torch.where(
        mask_index,
        x0_p,
        torch.tensor(-np.inf, device=x0.device)) # BxS

    # Select tokens to transfer based on confidence
    # Replace top-k most confident masked tokens
    for b in range(B):
        num_tokens = num_transfer_tokens[b, step_idx].item()
        if num_tokens > 0:
            _, select_indices = torch.topk(confidence[b], k=num_tokens)
            x[b, select_indices] = x0[b, select_indices]

    return x
