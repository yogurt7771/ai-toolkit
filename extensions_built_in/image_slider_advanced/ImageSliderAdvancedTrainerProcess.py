import copy
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn.functional as F

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.stable_diffusion_model import StableDiffusion
from toolkit.train_tools import get_torch_dtype, apply_snr_weight
from toolkit.prompt_utils import PromptEmbeds


class ImageSliderAdvancedConfig:
    def __init__(self, **kwargs):
        # --- auxiliary losses (Concept Slider Advanced style) ---
        # cap aux losses
        self.cap_ratio: float = float(kwargs.get("cap_ratio", 0.25))
        self.cap_main_floor: float = float(kwargs.get("cap_main_floor", 1e-4))
        self.cap_eps: float = float(kwargs.get("cap_eps", 1e-5))
        # linearity loss
        self.linearity_weight: float = float(kwargs.get("linearity_weight", 0.0))
        self.linearity_delta: float = float(kwargs.get("linearity_delta", 0.2))
        self.linearity_loss_type: str = str(kwargs.get("linearity_loss_type", "smooth_l1"))
        self.linearity_gate_threshold: float = float(kwargs.get("linearity_gate_threshold", 0.01))
        self.linearity_beta: float = float(kwargs.get("linearity_beta", 0.05))


class ImageSliderAdvancedTrainerProcess(DiffusionTrainer):
    sd: StableDiffusion

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.device = self.get_conf("device", self.job.device)
        self.device_torch = torch.device(self.device)
        self.slider_config = ImageSliderAdvancedConfig(**self.get_conf("slider", {}))
        # Force SDTrainer to route into `get_guided_loss`.
        self.do_guided_loss = True

    def _set_network_multiplier(self, m_list):
        if hasattr(self.network, "set_multiplier"):
            self.network.set_multiplier(m_list)
        else:
            self.network.multiplier = m_list

    def cap_aux(
        self,
        aux: torch.Tensor,
        main: torch.Tensor,
        *,
        ratio: float,
        main_floor: float,
        eps: float,
    ) -> torch.Tensor:
        with torch.no_grad():
            m = main.detach().clamp_min(main_floor)
            a = aux.detach()
            lam = (m * ratio) / (a + eps)
            lam = lam.clamp(0.0, 1.0)
        return aux * lam.detach()

    def smooth_l1_tensor_beta(self, x: torch.Tensor, beta: float, eps: float = 1e-5):
        absx = x.abs()
        beta_target = torch.quantile(absx.detach().float(), q=beta).clamp_min(eps)
        return torch.where(
            absx < beta_target, 0.5 * (absx**2) / beta_target, absx - 0.5 * beta_target
        )

    def build_pair_diff_mask(
        self,
        pos_latents: torch.Tensor,
        neg_latents: torch.Tensor,
        blur_ks: int = 5,  # 0 表示不平滑
        q_low: float = 0.01,  # 分位数替代 min
        q_high: float = 0.99,  # 分位数替代 max
        tau: float = 0.2,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Returns (B,1,H,W) in [0,1], detached.
        """
        with torch.no_grad():
            # 用 fp32 计算更稳
            d = (
                (pos_latents.float() - neg_latents.float()).abs().mean(dim=1, keepdim=True)
            )  # (B,1,H,W)

            if blur_ks and blur_ks > 1:
                pad = blur_ks // 2
                d = F.avg_pool2d(d, kernel_size=blur_ks, stride=1, padding=pad)

            flat = d.flatten(start_dim=2)  # (B,1,HW)
            lo = torch.quantile(flat, q_low, dim=2, keepdim=True).view(d.size(0), 1, 1, 1)
            hi = torch.quantile(flat, q_high, dim=2, keepdim=True).view(d.size(0), 1, 1, 1)

            denom = (hi - lo).clamp_min(eps)
            m = ((d - lo) / denom).clamp(0.0, 1.0)
            m = ((m - tau) / (1.0 - tau + eps)).clamp(0.0, 1.0)

        return m.detach()

    def get_linearity_loss(
        self,
        noisy_latents: torch.Tensor,
        embeds,
        timesteps: torch.Tensor,
        batch: DataLoaderBatchDTO,
        class_pred: torch.Tensor,
        multipliers: list,
        weight: float = 1.0,
        delta: float = 0.1,
        loss_type: str = "smooth_l1",
        gate_threshold: float = 0.01,
        beta: float = 0.05,
    ) -> torch.Tensor:
        """
        Linearity loss around current multiplier list.

        We compute mid prediction with `m-delta` and `m+delta` under no_grad,
        then penalize deviation of `class_pred(m)` from the midpoint (with grads through class_pred only).

        If `loss_mask` is provided (B,1,H,W) or (B,C,H,W), we compute a masked/weighted spatial mean
        so the regularizer focuses on the changed region.
        """
        was_unet_training = self.sd.unet.training
        old_m = self.network.multiplier

        # per-sample time weighting (ConceptSliderTrainer style)
        t = timesteps.to(device=class_pred.device, dtype=class_pred.dtype)
        linearity_weight = (weight * (t / 1000.0) ** 2).detach()

        with torch.no_grad():
            self.sd.unet.eval()

            m_left = [m - delta for m in multipliers]
            self._set_network_multiplier(m_left)
            pred_left = self.sd.predict_noise(
                latents=noisy_latents.to(self.device_torch, dtype=class_pred.dtype).detach(),
                conditional_embeddings=embeds.to(self.device_torch, dtype=class_pred.dtype).detach(),
                timestep=timesteps,
                guidance_scale=1.0,
                guidance_embedding_scale=1.0,
                batch=batch,
            ).detach()

            m_right = [m + delta for m in multipliers]
            self._set_network_multiplier(m_right)
            pred_right = self.sd.predict_noise(
                latents=noisy_latents.to(self.device_torch, dtype=class_pred.dtype).detach(),
                conditional_embeddings=embeds.to(self.device_torch, dtype=class_pred.dtype).detach(),
                timestep=timesteps,
                guidance_scale=1.0,
                guidance_embedding_scale=1.0,
                batch=batch,
            ).detach()

        # restore
        if was_unet_training:
            self.sd.unet.train()
        self._set_network_multiplier(old_m)

        mid = (pred_left + pred_right) / 2.0

        if loss_type == "mae":
            loss_map = F.l1_loss(class_pred, mid, reduction="none")
        elif loss_type == "mse":
            loss_map = F.mse_loss(class_pred, mid, reduction="none")
        elif loss_type == "smooth_l1":
            diff = class_pred - mid
            loss_map = self.smooth_l1_tensor_beta(diff, beta)
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")

        per = loss_map.flatten(start_dim=1).mean(dim=1)
        thr = torch.as_tensor(gate_threshold, device=per.device, dtype=per.dtype)
        per = F.relu(per - thr)
        return (per * linearity_weight)

    def _get_loss_target(
        self,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        batch: DataLoaderBatchDTO,
    ) -> torch.Tensor:
        # Follow SDTrainer.calculate_loss target selection (minimal subset).
        if hasattr(self.sd, "get_loss_target"):
            return self.sd.get_loss_target(
                noise=noise,
                batch=batch,
                timesteps=timesteps,
            ).detach()
        if getattr(self.sd, "is_flow_matching", False):
            return (noise - batch.latents).detach()
        if self.sd.prediction_type == "v_prediction" and hasattr(self.sd.noise_scheduler, "get_velocity"):
            return self.sd.noise_scheduler.get_velocity(batch.latents, noise, timesteps).detach()
        return noise.detach()

    def get_keep_loss(
        self,
        noisy_latents: torch.Tensor,
        conditional_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        batch: DataLoaderBatchDTO,
        pred_kwargs: dict,
        class_pred: torch.Tensor,
        mc: torch.Tensor,
        eps: float = 1e-6,
    ):
        was_unet_training = self.sd.unet.training
        was_network_active = False
        if self.network is not None:
            was_network_active = self.network.is_active
            self.network.is_active = False

        self.sd.unet.eval()
        with torch.no_grad():
            pred_baseline = self.sd.predict_noise(
                latents=noisy_latents.to(self.device_torch, dtype=self.sd.torch_dtype).detach(),
                conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=self.sd.torch_dtype).detach(),
                timestep=timesteps,
                guidance_scale=1.0,
                guidance_embedding_scale=1.0,
                batch=batch,
                **pred_kwargs,
            ).detach()
        if was_unet_training:
            self.sd.unet.train()
        if self.network is not None:
            self.network.is_active = was_network_active

        inv = 1.0 - mc
        err_keep = (class_pred - pred_baseline) ** 2

        keep_sum = (err_keep * inv).sum(dim=[1,2,3])
        keep_den = inv.sum(dim=[1,2,3]).clamp_min(eps)
        keep_mean = keep_sum / keep_den
        return keep_mean

    def get_edit_loss(
        self,
        target: torch.Tensor,
        class_pred: torch.Tensor,
        mc: torch.Tensor,
        eps: float = 1e-6,
    ):
        err = (class_pred - target) ** 2
        edit_sum = (err * mc).sum(dim=[1,2,3])
        edit_den = mc.sum(dim=[1,2,3]).clamp_min(eps)
        edit_mean = edit_sum / edit_den
        return edit_mean

    def get_direct_loss(
        self,
        noisy_latents: torch.Tensor,
        conditional_embeds: torch.Tensor,
        timesteps: torch.Tensor,
        batch: DataLoaderBatchDTO,
        noise: torch.Tensor,
        loss_mask: torch.Tensor | None,
        multipliers: list,
        **pred_kwargs: dict,
    ):
        self._set_network_multiplier(list(multipliers))
        pred = self.sd.predict_noise(
            latents=noisy_latents.to(self.device_torch, dtype=self.sd.torch_dtype).detach(),
            conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=self.sd.torch_dtype).detach(),
            timestep=timesteps,
            guidance_scale=1.0,
            guidance_embedding_scale=1.0,
            batch=batch,
            **pred_kwargs,
        )
        target = self._get_loss_target(
            noise=noise,
            timesteps=timesteps,
            batch=batch,
        )
        if loss_mask is not None:
            m = loss_mask.to(device=pred.device, dtype=pred.dtype)
        else:
            m = torch.ones(
                (pred.shape[0], 1, pred.shape[2], pred.shape[3]),
                device=pred.device,
                dtype=pred.dtype,
            )
        tau = 0.1
        m = ((m - tau) / (1.0 - tau + 1e-6)).clamp(0.0, 1.0)  # (B,1,H,W)
        mc = m.expand(-1, pred.shape[1], -1, -1)

        edit_loss_per = self.get_edit_loss(
            target=target,
            class_pred=pred,
            mc=mc,
        )

        keep_loss_per = self.get_keep_loss(
            noisy_latents=noisy_latents,
            conditional_embeds=conditional_embeds,
            timesteps=timesteps,
            batch=batch,
            pred_kwargs=pred_kwargs,
            class_pred=pred,
            mc=mc,
        )

        keep_cap_per = self.cap_aux(keep_loss_per, edit_loss_per, ratio=0.5, main_floor=self.slider_config.cap_main_floor, eps=self.slider_config.cap_eps)

        loss_main_per = edit_loss_per + keep_loss_per
        loss_main = loss_main_per.mean()

        linearity_loss_per = self.get_linearity_loss(
            noisy_latents=noisy_latents,
            embeds=conditional_embeds,
            timesteps=timesteps,
            batch=batch,
            class_pred=pred,
            multipliers=multipliers,
            weight=self.slider_config.linearity_weight,
            delta=self.slider_config.linearity_delta,
            loss_type=self.slider_config.linearity_loss_type,
            gate_threshold=self.slider_config.linearity_gate_threshold,
            beta=self.slider_config.linearity_beta,
        )
        linearity_loss_per = self.cap_aux(
            linearity_loss_per,
            loss_main_per,
            ratio=self.slider_config.cap_ratio,
            main_floor=self.slider_config.cap_main_floor,
            eps=self.slider_config.cap_eps,
        )
        linearity_loss = linearity_loss_per.mean()
        loss_total = loss_main + linearity_loss

        loss_total.backward()
        loss_total = loss_total.detach()
        return loss_total

    def get_guided_loss(
        self,
        noisy_latents: torch.Tensor,
        conditional_embeds: PromptEmbeds,
        match_adapter_assist: bool,
        network_weight_list: list,
        timesteps: torch.Tensor,
        pred_kwargs: dict,
        batch: "DataLoaderBatchDTO",
        noise: torch.Tensor,
        unconditional_embeds: Optional[PromptEmbeds] = None,
        **kwargs,
    ):
        """
        Image slider: treat `datasets[].unconditional_path` as the paired (negative) image.

        IMPORTANT (gradient checkpointing):
        We must NOT switch network multipliers inside a single "loss graph" that will be replayed by
        checkpointing on backward. So we do two separate passes like `concept_slider_advanced`:
        - POS: multiplier = +network_weight_list, forward -> backward
        - NEG: multiplier = -network_weight_list, forward -> backward
        Then return a detached loss with `requires_grad_(True)` so SDTrainer can safely call backward again.
        """
        if self.network is None:
            raise ValueError("image_slider_advanced requires a LoRA network (network_config).")
        if batch.unconditional_latents is None:
            raise ValueError(
                "`image_slider_advanced` requires paired images via `datasets[].unconditional_path` "
                "so `batch.unconditional_latents` is available."
            )

        dtype = get_torch_dtype(self.train_config.dtype)
        device = self.device_torch

        # Inputs are already mostly detached by SDTrainer; keep it explicit.
        conditional_noisy = noisy_latents.to(device, dtype=dtype).detach()
        noise = noise.to(device, dtype=dtype).detach()
        timesteps = timesteps.to(device)

        conditional_latents = batch.latents.to(device, dtype=dtype).detach()
        unconditional_latents = batch.unconditional_latents.to(device, dtype=dtype).detach()

        bsz = conditional_noisy.shape[0]
        if conditional_latents.shape[0] != bsz:
            if conditional_latents.shape[0] * 2 == bsz:
                conditional_latents = torch.cat([conditional_latents, conditional_latents], dim=0)
            else:
                raise ValueError(
                    f"conditional_latents batch mismatch: got {conditional_latents.shape[0]}, expected {bsz}."
                )
        if unconditional_latents.shape[0] != bsz:
            if unconditional_latents.shape[0] * 2 == bsz:
                unconditional_latents = torch.cat([unconditional_latents, unconditional_latents], dim=0)
            else:
                raise ValueError(
                    f"unconditional_latents batch mismatch: got {unconditional_latents.shape[0]}, expected {bsz}."
                )

        # Build a neg-batch view where `batch.latents` points at the unconditional/pair latents.
        batch_pos = batch
        batch_neg = copy.copy(batch)
        batch_neg.latents = unconditional_latents

        # Build a spatial diff mask once (shared by both passes).
        # Shape: (B,1,H,W) then expanded to channels when applying.
        with torch.no_grad():
            diff_mask = self.build_pair_diff_mask(
                pos_latents=conditional_latents,
                neg_latents=unconditional_latents,
            )

        # get noisy latents for the neg/pair image and condition them with the same controls/masks/etc
        unconditional_noisy = self.sd.add_noise(unconditional_latents, noise, timesteps).detach()
        unconditional_noisy = self.sd.condition_noisy_latents(unconditional_noisy, batch_neg)
        if self.adapter is not None and hasattr(self.adapter, "condition_noisy_latents"):
            unconditional_noisy = self.adapter.condition_noisy_latents(unconditional_noisy, batch_neg)

        # Set per-sample multipliers for polarity (+ for pos, - for neg).
        pos_mult = [weight * 1.0 for weight in network_weight_list]
        neg_mult = [weight * -1.0 for weight in network_weight_list]
        old_multiplier = self.network.multiplier

        loss_pos_total = self.get_direct_loss(
            noisy_latents=conditional_noisy,
            conditional_embeds=conditional_embeds,
            timesteps=timesteps,
            batch=batch_pos,
            noise=noise,
            loss_mask=diff_mask,
            multipliers=pos_mult,
            **pred_kwargs,
        )
        loss_neg_total = self.get_direct_loss(
            noisy_latents=unconditional_noisy,
            conditional_embeds=conditional_embeds,
            timesteps=timesteps,
            batch=batch_neg,
            noise=noise,
            loss_mask=diff_mask,
            multipliers=neg_mult,
            **pred_kwargs,
        )

        # restore network multiplier for parent code
        self._set_network_multiplier(old_multiplier)

        # detach it so parent class can run backward without error
        total_loss = (loss_pos_total + loss_neg_total) / 2.0
        total_loss.requires_grad_(True)
        return total_loss
