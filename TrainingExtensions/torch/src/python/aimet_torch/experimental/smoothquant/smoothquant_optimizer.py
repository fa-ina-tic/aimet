"""Optimizer for Smoothquant"""

import torch
import functools
from aimet_torch.v2.quantsim import QuantizationSimModel
from typing import Dict
from tqdm import tqdm
import itertools
import time
from pathlib import Path
import os

from aimet_torch.common.utils import AimetLogger
from torch.utils.data import DataLoader
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm

SMOOTHQUANT_ARTIFACT_DIR = "./aimet_smoothquant_artifact/"

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

class Smoothquant:
    @classmethod
    def apply_smoothquant(
        cls,
        quant_sim: QuantizationSimModel,
        dataloader: DataLoader,
        alpha: float = 0.5,
        num_iterations: int = 512,
        output_path: str = SMOOTHQUANT_ARTIFACT_DIR
    ):
        output_path = Path(output_path)
        os.makedirs(output_path, exist_ok=True)

        start_smq_optmztn_time = time.perf_counter()
        act_scales = cls._get_act_scales(
            quant_sim,
            dataloader,
            num_iterations,
        )
        cls._smooth_lm(
            quant_sim,
            act_scales,
            alpha,
            output_path
        )
        
        total_smq_optmztn_time = time.perf_counter() - start_smq_optmztn_time
        _logger.info("Took %.4f seconds for smq optimization ", total_smq_optmztn_time)
        return None
    
    @classmethod
    def _get_act_scales(
        cls,
        quant_sim: QuantizationSimModel,
        dataloader: DataLoader,
        num_iterations: int
    ) -> Dict:
        device = quant_sim.model.device
        act_scales = {}

        def stat_tensor(name, tensor):
            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).abs().detach()
            comming_max = torch.max(tensor, dim=0)[0].float().cpu()
            if name in act_scales:
                act_scales[name] = torch.max(act_scales[name], comming_max)
            else:
                act_scales[name] = comming_max

        def stat_input_hook(m, x, y, name):
            if isinstance(x, tuple):
                x = x[0]
            stat_tensor(name, x)

        hooks = []
        for name, module in quant_sim.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                hooks.append(
                    module.register_forward_hook(functools.partial(stat_input_hook, name=name))
                )
        
        ### Iteration through dataset
        for text, _ in itertools.islice(dataloader, num_iterations):
            _ = quant_sim.model(text.to(device))

        for h in hooks:
            h.remove()

        return act_scales
    
    @torch.no_grad()
    @classmethod
    def _smooth_lm(quant_sim, scales, alpha=0.5):

        def _smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
            if not isinstance(fcs, list):
                fcs = [fcs]
            assert isinstance(ln, (LlamaRMSNorm))
            for fc in fcs:
                assert isinstance(fc, torch.nn.Linear)
                assert ln.weight.numel() == fc.in_features == act_scales.numel()
            device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
            act_scales = act_scales.to(device=device, dtype=dtype)
            weight_scales = torch.cat(
                [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
            )
            weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
            scales = (
                (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
                .clamp(min=1e-5)
                .to(device)
                .to(dtype)
            )

            ln.weight.div_(scales)
            for fc in fcs:
                fc.weight.mul_(scales.view(1, -1))

        for name, module in quant_sim.model.named_modules():
            if isinstance(module, LlamaDecoderLayer):
                attn_ln = module.input_layernorm
                qkv = [
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ]

                qkv_input_scales = scales[name + ".self_attn.q_proj"]
                _smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

                ffn_ln = module.post_attention_layernorm # feed forward norm
                fcs  = [module.mlp.gate_proj, module.mlp.up_proj]
                fcs_input_scale = scales[name + ".mlp.gate_proj"]
                _smooth_ln_fcs(ffn_ln, fcs, fcs_input_scale, alpha)

        return None

apply_smoothquant = Smoothquant.apply_smoothquant
