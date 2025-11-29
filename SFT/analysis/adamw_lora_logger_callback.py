import wandb
import torch
import matplotlib.pyplot as plt
import torch.distributed as dist
from transformers import TrainerCallback
from io import BytesIO
from PIL import Image

import os.path as osp
from pathlib import Path

import numpy as np

class AdamWLoRAMonitorCallback(TrainerCallback):

    def __init__(self,
            logging_steps: int=2,
            save_steps: int=100,
            output_dir: str="logs",
            local_rank: int=0,
            topk: int=5,
            debugging: bool=False,
        ):
        self.topk = topk
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
        else:
            self.rank = local_rank
        self.rank_zero = (self.rank == 0)
        self.output_dir = output_dir
        self.debugging = debugging

        self.local_rank = local_rank

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_log(self, args, state, control, **kwargs):

        model = kwargs["model"]
        step = state.global_step
        logs = {}
        saved_state = {}

        if (step + 1) % self.logging_steps != 0:
            return

        self.optimizer = self.trainer.optimizer
        if self.optimizer is not None:
            self.eps = self.optimizer.param_groups[0].get("eps", 1e-8)

        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                if not param.requires_grad:
                    continue

                state_dict = self.optimizer.state.get(param, None)
                if state_dict is None or 'exp_avg' not in state_dict or 'exp_avg_sq' not in state_dict:
                    continue

                # print(f"Track at {self.local_rank}: {name}")

                # exp_avg and exp_avg_sq computed as https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py#L442-L460
                m_t = state_dict['exp_avg'].detach().cpu() # LoRA-A: [r, dim]; LoRA-B: [dim, r]
                v_t = state_dict['exp_avg_sq'].detach().cpu() # LoRA-A: [r, dim]; LoRA-B: [dim, r]
                optim_step = state_dict['step']

                update = m_t / (v_t.sqrt() + self.eps)

                logs.update({
                    f"{name}/m_norm": m_t.norm().item(),
                    f"{name}/m_std": m_t.std().item(),
                    f"{name}/v_mean": v_t.mean().item(),
                    f"{name}/v_std": v_t.std().item(),
                    f"{name}/update_mean": update.mean().item(),
                    f"{name}/update_std": update.std().item(),
                    f"{name}/update_norm": update.norm().item(),
                    f"{name}/topk_mean_row": update.abs().sum(dim=1).topk(self.topk).values.mean().item(),
                    f"{name}/topk_mean_col": update.abs().sum(dim=0).topk(self.topk).values.mean().item(),
                })

                param_state = {}
                for key, value in state_dict.items():
                    if isinstance(value, torch.Tensor):
                        param_state[key] = value.detach().cpu()
                    else:
                        param_state[key] = value

                if "epsilon" not in param_state:
                    param_state["epsilon"] = self.eps

                saved_state[name] = param_state

        if logs and self.rank_zero and not self.debugging:
            wandb.log(logs)

        probe_step = step + 1

        if probe_step % self.save_steps == 0 and saved_state:
            save_dir = osp.join(self.output_dir, f"checkpoint-{probe_step}", f"rank-{self.rank}")
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            save_path = osp.join(save_dir, f"lora_optim_step{probe_step}-rank{self.rank}.pt")
            torch.save(saved_state, save_path)
