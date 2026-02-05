import copy
from collections import OrderedDict

import torch
import torch.nn.functional as F

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.stable_diffusion_model import StableDiffusion
from toolkit.train_tools import get_torch_dtype, apply_snr_weight


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
        *,
        pos_latents: torch.Tensor,
        neg_latents: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build a per-sample spatial weight map from paired clean latents:
        m = |pos-neg| normalized to [0,1] by per-sample, per-channel min/max (over spatial dims).

        Returns (B,C,H,W), detached.
        """
        d = (pos_latents - neg_latents).abs()  # (B,C,H,W)
        flat = d.flatten(start_dim=2)  # (B,C,HW)
        d_min = flat.min(dim=2, keepdim=True).values.view(d.shape[0], d.shape[1], 1, 1)
        d_max = flat.max(dim=2, keepdim=True).values.view(d.shape[0], d.shape[1], 1, 1)
        m = (d - d_min) / (d_max - d_min + 1e-6)
        m = m.clamp(0.0, 1.0)
        return m.to(dtype=dtype).detach()

    def get_linearity_loss(
        self,
        *,
        noisy_latents: torch.Tensor,
        embeds,
        timesteps: torch.Tensor,
        batch: DataLoaderBatchDTO,
        class_pred: torch.Tensor,
        loss_mask: torch.Tensor | None,
        m_list: list,
        weight: float,
        delta: float,
        loss_type: str,
        gate_threshold: float,
        beta: float,
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

            m_left = [m - delta for m in m_list]
            self._set_network_multiplier(m_left)
            pred_left = self.sd.predict_noise(
                latents=noisy_latents.to(self.device_torch, dtype=class_pred.dtype).detach(),
                conditional_embeddings=embeds.to(self.device_torch, dtype=class_pred.dtype).detach(),
                timestep=timesteps,
                guidance_scale=1.0,
                guidance_embedding_scale=1.0,
                batch=batch,
            ).detach()

            m_right = [m + delta for m in m_list]
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

        if loss_mask is not None:
            m = loss_mask
            if m.ndim != 4:
                raise ValueError(f"loss_mask must be 4D (B,1,H,W) or (B,C,H,W), got {tuple(m.shape)}")
            if m.shape[1] == 1 and loss_map.shape[1] != 1:
                m = m.expand(-1, loss_map.shape[1], -1, -1)
            elif m.shape[1] != loss_map.shape[1]:
                raise ValueError(
                    f"loss_mask channel mismatch: mask has {m.shape[1]}, loss_map has {loss_map.shape[1]}"
                )
            m = m.to(device=loss_map.device, dtype=loss_map.dtype).detach()
            denom = m.sum(dim=(1, 2, 3)).clamp_min(1e-6)
            per = (loss_map * m).sum(dim=(1, 2, 3)) / denom
        else:
            per = loss_map.flatten(start_dim=1).mean(dim=1)

        thr = torch.as_tensor(gate_threshold, device=per.device, dtype=per.dtype)
        per = F.relu(per - thr)
        return (per * linearity_weight).mean()

    def _get_loss_target(
        self,
        *,
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

    def get_guided_loss(
        self,
        noisy_latents: torch.Tensor,
        conditional_embeds,
        match_adapter_assist: bool,
        network_weight_list: list,
        timesteps: torch.Tensor,
        pred_kwargs: dict,
        batch: "DataLoaderBatchDTO",
        noise: torch.Tensor,
        unconditional_embeds=None,
        mask_multiplier=None,
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
                dtype=dtype,
            )
        linearity_mask = diff_mask
        if mask_multiplier is not None:
            # mask_multiplier is typically (B,C,H,W); multiply channel-wise when possible.
            linearity_mask = linearity_mask * mask_multiplier.to(device=device, dtype=dtype)

        # get noisy latents for the neg/pair image and condition them with the same controls/masks/etc
        unconditional_noisy = self.sd.add_noise(unconditional_latents, noise, timesteps).detach()
        unconditional_noisy = self.sd.condition_noisy_latents(unconditional_noisy, batch_neg)
        if self.adapter is not None and hasattr(self.adapter, "condition_noisy_latents"):
            unconditional_noisy = self.adapter.condition_noisy_latents(unconditional_noisy, batch_neg)

        # Set per-sample multipliers for polarity (+ for pos, - for neg).
        pos_mult = list(network_weight_list)
        neg_mult = list(network_weight_list)
        old_multiplier = self.network.multiplier

        # -----------------------
        # POS pass (train +weight)
        # -----------------------
        self._set_network_multiplier(list(pos_mult))
        pred_pos = self.sd.predict_noise(
            latents=conditional_noisy.to(device, dtype=dtype).detach(),
            conditional_embeddings=conditional_embeds.to(device, dtype=dtype).detach(),
            timestep=timesteps,
            guidance_scale=1.0,
            guidance_embedding_scale=1.0,
            batch=batch_pos,
            **pred_kwargs,
        )
        target_pos = self._get_loss_target(
            noise=noise,
            timesteps=timesteps,
            batch=batch_pos,
        )
        m = diff_mask
        if m.shape[1] == 1 and pred_pos.shape[1] != 1:
            m = m.expand(-1, pred_pos.shape[1], -1, -1)
        if mask_multiplier is not None:
            m = m * mask_multiplier.to(device=device, dtype=dtype)
        loss_pos_vec = ((pred_pos.float() - target_pos.float()) ** 2) * m.float()
        loss_pos_vec = loss_pos_vec.mean([1, 2, 3])
        if self.train_config.min_snr_gamma is not None and self.train_config.min_snr_gamma > 1e-6:
            loss_pos_vec = apply_snr_weight(loss_pos_vec, timesteps, self.sd.noise_scheduler, self.train_config.min_snr_gamma)
        loss_pos_main = loss_pos_vec.mean()

        loss_pos_total = loss_pos_main
        if self.slider_config.linearity_weight > 1e-6:
            lin_pos = self.get_linearity_loss(
                noisy_latents=conditional_noisy,
                embeds=conditional_embeds,
                timesteps=timesteps,
                batch=batch_pos,
                class_pred=pred_pos,
                loss_mask=linearity_mask,
                m_list=list(pos_mult),
                weight=self.slider_config.linearity_weight,
                delta=self.slider_config.linearity_delta,
                loss_type=self.slider_config.linearity_loss_type,
                gate_threshold=self.slider_config.linearity_gate_threshold,
                beta=self.slider_config.linearity_beta,
            )
            lin_pos = self.cap_aux(
                lin_pos,
                loss_pos_main,
                ratio=self.slider_config.cap_ratio,
                main_floor=self.slider_config.cap_main_floor,
                eps=self.slider_config.cap_eps,
            )
            loss_pos_total = loss_pos_total + lin_pos

        self.accelerator.backward(loss_pos_total)

        # -----------------------
        # NEG pass (train -weight)
        # -----------------------
        neg_multiplier = [(-w) for w in neg_mult]
        self._set_network_multiplier(neg_multiplier)
        pred_neg = self.sd.predict_noise(
            latents=unconditional_noisy.to(device, dtype=dtype).detach(),
            conditional_embeddings=conditional_embeds.to(device, dtype=dtype).detach(),
            timestep=timesteps,
            guidance_scale=1.0,
            guidance_embedding_scale=1.0,
            batch=batch_neg,
            **pred_kwargs,
        )
        target_neg = self._get_loss_target(
            noise=noise,
            timesteps=timesteps,
            batch=batch_neg,
        )
        m = diff_mask
        if m.shape[1] == 1 and pred_neg.shape[1] != 1:
            m = m.expand(-1, pred_neg.shape[1], -1, -1)
        if mask_multiplier is not None:
            m = m * mask_multiplier.to(device=device, dtype=dtype)
        loss_neg_vec = ((pred_neg.float() - target_neg.float()) ** 2) * m.float()
        loss_neg_vec = loss_neg_vec.mean([1, 2, 3])
        if self.train_config.min_snr_gamma is not None and self.train_config.min_snr_gamma > 1e-6:
            loss_neg_vec = apply_snr_weight(loss_neg_vec, timesteps, self.sd.noise_scheduler, self.train_config.min_snr_gamma)
        loss_neg_main = loss_neg_vec.mean()

        loss_neg_total = loss_neg_main
        if self.slider_config.linearity_weight > 1e-6:
            lin_neg = self.get_linearity_loss(
                noisy_latents=unconditional_noisy,
                embeds=conditional_embeds,
                timesteps=timesteps,
                batch=batch_neg,
                class_pred=pred_neg,
                loss_mask=linearity_mask,
                m_list=neg_multiplier,
                weight=self.slider_config.linearity_weight,
                delta=self.slider_config.linearity_delta,
                loss_type=self.slider_config.linearity_loss_type,
                gate_threshold=self.slider_config.linearity_gate_threshold,
                beta=self.slider_config.linearity_beta,
            )
            lin_neg = self.cap_aux(
                lin_neg,
                loss_neg_main,
                ratio=self.slider_config.cap_ratio,
                main_floor=self.slider_config.cap_main_floor,
                eps=self.slider_config.cap_eps,
            )
            loss_neg_total = loss_neg_total + lin_neg

        self.accelerator.backward(loss_neg_total)

        # restore network multiplier for parent code
        self._set_network_multiplier(old_multiplier)

        # detach it so parent class can run backward without error
        total_loss = (loss_pos_total.detach() + loss_neg_total.detach()) / 2.0
        total_loss.requires_grad_(True)
        return total_loss
