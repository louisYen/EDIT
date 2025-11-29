import os
import re
from collections import OrderedDict
from typing import Dict, Literal

import os.path as osp
from pathlib import Path

import torch
from natsort import natsorted
from tqdm import tqdm

def get_step_from_ckpt(ckpt_dir: str):
    match = re.search(r"checkpoint-(\d+)", ckpt_dir)
    if match:
        return int(match.group(1))
    else:
        return None

def extract_step_from_path(checkpoint_path: str) -> int:
    match = re.search(r"checkpoint-(\d+)", str(checkpoint_path))
    if not match:
        raise ValueError(f"Cannot extract step from path: {checkpoint_path}")
    return int(match.group(1))


def remove_default_suffix(state_dict: Dict) -> Dict:
    """
    Remove '.default' suffix in parameter keys if present.
    """
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace(".default", "")
        new_state_dict[new_key] = value
    return new_state_dict


def _load_state(path: str) -> Dict:
    if not osp.isfile(path):
        raise FileNotFoundError(f"Optimizer state file not found: {path}")
    return torch.load(path, map_location="cpu")

def _accumulate_sum(
    agg_sums: Dict[str, torch.Tensor],
    counts: Dict[str, int],
    layer_name: str,
    update: torch.Tensor,
):
    if layer_name not in agg_sums:
        agg_sums[layer_name] = update.clone()
        counts[layer_name] = 1
    else:
        agg_sums[layer_name] += update
        counts[layer_name] += 1

def collect_optim_states(
    log_path: str,
    start_step: int = 2,
    end_step: int = 200,
    multi_rank_reduce: Literal["auto", "rank0", "average_update"] = "auto",
) -> Dict[str, torch.Tensor]:

    # Find all checkpoint-* dirs
    checkpoint_dirs = natsorted(
        [
            osp.join(log_path, d)
            for d in os.listdir(log_path)
            if d.startswith("checkpoint-") and osp.isdir(osp.join(log_path, d))
        ]
    )

    # Filter by step range
    checkpoint_dirs = [
        d
        for d in checkpoint_dirs
        if (step := get_step_from_ckpt(d)) is not None
        and start_step <= step <= end_step
    ]

    if not checkpoint_dirs:
        raise RuntimeError(
            f"No checkpoint-* dirs in {log_path} within [{start_step}, {end_step}]"
        )

    agg_sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}

    for ckpt_dir in tqdm(checkpoint_dirs, ncols=100, desc="Read checkpoints"):
        step = get_step_from_ckpt(ckpt_dir)

        # detect per-rank layout
        rank_subdirs = [
            d for d in os.listdir(ckpt_dir)
            if d.startswith("rank-") and osp.isdir(osp.join(ckpt_dir, d))
        ]

        # ---------------- Single-file layout ----------------
        if not rank_subdirs:
            optim_path = osp.join(ckpt_dir, f"lora_optim_step{step}.pt")
            optim_data = _load_state(optim_path)
            # optim_data = remove_default_suffix(optim_data)

            for key, entry in optim_data.items():
                if not isinstance(entry, dict):
                    continue
                if "exp_avg" not in entry or "exp_avg_sq" not in entry:
                    continue

                m_t = entry["exp_avg"].float()
                v_t = entry["exp_avg_sq"].float()
                eps = float(entry.get("epsilon", 1e-8))
                update = m_t / (v_t.sqrt() + eps)

                _accumulate_sum(agg_sums, counts, key, update)

            continue

        # ---------------- Multi-rank layout ----------------
        rank_subdirs = natsorted(rank_subdirs)
        mode = "average_update" if multi_rank_reduce == "auto" else multi_rank_reduce

        per_rank_states = []
        for rdir in rank_subdirs:
            rank_id = int(rdir.split("-")[-1])  # "rank-0" -> 0

            cand1 = osp.join(
                ckpt_dir, rdir, f"lora_optim_step{step}-rank{rank_id}.pt"
            )
            cand2 = osp.join(
                ckpt_dir, rdir, f"lora_optim_step{step}.pt"
            )

            if osp.isfile(cand1):
                p = cand1
            elif osp.isfile(cand2):
                p = cand2
            else:
                raise FileNotFoundError(
                    f"Missing optimizer state for rank {rank_id} in {ckpt_dir}"
                )

            state = _load_state(p)
            # state = remove_default_suffix(state)
            per_rank_states.append(state)

        ref = per_rank_states[0]
        layer_keys = list(ref.keys())

        for key in layer_keys:
            # ensure layer is present on all ranks
            if any(key not in rs for rs in per_rank_states):
                continue

            rank_updates = []

            for rs in per_rank_states:
                entry = rs[key]
                if "exp_avg" not in entry or "exp_avg_sq" not in entry:
                    rank_updates = []
                    break

                m_t = entry["exp_avg"].float()
                v_t = entry["exp_avg_sq"].float()
                eps = float(entry.get("epsilon", 1e-8))

                if m_t.shape != v_t.shape:
                    rank_updates = []
                    break

                update = m_t / (v_t.sqrt() + eps)
                rank_updates.append(update)

            if not rank_updates:
                continue

            if mode == "rank0":
                avg_update = rank_updates[0]
            else:  # "average_update"
                stacked = torch.stack(rank_updates, dim=0)  # (R, ...)
                avg_update = stacked.mean(dim=0)

            _accumulate_sum(agg_sums, counts, key, avg_update)

    # Final mean per layer: sum / count
    mean_updates: Dict[str, torch.Tensor] = {}
    for key, s in agg_sums.items():
        mean_updates[key] = s / counts[key]

    return mean_updates

def load_or_collect_optim_states(
    checkpoint_path: str,
    multi_rank_reduce: Literal["auto", "rank0", "average_update"] = "auto",
) -> Dict[str, torch.Tensor]:
    """
    Either load a cached aggregated optimizer state (mean updates), or
    collect & aggregate it and then cache the result.

    Args:
        checkpoint_path:
            Path to any file inside a checkpoint folder, or the checkpoint
            folder itself (must contain 'checkpoint-<step>').
        multi_rank_reduce:
            "auto"           -> if rank-* dirs exist, average updates across ranks
            "rank0"          -> only use rank-0 files in multi-rank case
            "average_update" -> explicitly average per-rank updates

    Returns:
        optim_state:
            dict[layer_name] -> Tensor (mean update over all considered checkpoints)
    """
    ckpt_step = extract_step_from_path(checkpoint_path)
    log_root = Path(checkpoint_path).parents[0]

    cache_path = log_root / f"optim_state_mean.pt"

    if cache_path.exists():
        optim_state = torch.load(cache_path, map_location="cpu")
        print(f"Loaded optimizer state from cache: {cache_path}")
        return optim_state

    # Collect & aggregate optimizer states (mean over ranks & steps)
    mean_updates = collect_optim_states(
        log_path=str(log_root),
        start_step=2,
        end_step=ckpt_step,
        multi_rank_reduce=multi_rank_reduce,
    )

    torch.save(mean_updates, cache_path)
    print(f"Saved optimizer state to cache: {cache_path}")

    return mean_updates
