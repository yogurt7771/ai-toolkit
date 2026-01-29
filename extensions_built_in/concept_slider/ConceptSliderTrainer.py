from collections import OrderedDict
from typing import Optional

import torch

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.train_tools import get_torch_dtype


class ConceptSliderTrainerConfig:
    def __init__(self, **kwargs):
        self.guidance_strength: float = kwargs.get("guidance_strength", 3.0)
        self.anchor_strength: float = kwargs.get("anchor_strength", 1.0)
        self.positive_prompt: str = kwargs.get("positive_prompt", "")
        self.negative_prompt: str = kwargs.get("negative_prompt", "")
        self.neutral_prompt: str = kwargs.get("neutral_prompt", "")
        self.target_classes: list[str] = kwargs.get("target_class", [])
        self.anchor_class: Optional[str] = kwargs.get("anchor_class", None)
        self.multiplier: float = float(kwargs.get("multiplier", 1.0))
        self.linearity_weight: float = float(kwargs.get("linearity_weight", 0.1))
        self.linearity_delta: float = float(kwargs.get("linearity_delta", 0.1))
        self.linearity_loss_type: str = str(kwargs.get("linearity_loss_type", "smooth_l1"))
        self.linearity_huber_beta: float = float(kwargs.get("linearity_huber_beta", 0.1))


def norm_like_tensor(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Normalize the tensor to have the same mean and std as the target tensor."""
    tensor_mean = tensor.mean()
    tensor_std = tensor.std()
    target_mean = target.mean()
    target_std = target.std()
    normalized_tensor = (tensor - tensor_mean) / (
        tensor_std + 1e-8
    ) * target_std + target_mean
    return normalized_tensor


class ConceptSliderTrainer(DiffusionTrainer):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.do_guided_loss = True

        self.slider: ConceptSliderTrainerConfig = None
        self.positive_prompt_embeds: Optional[PromptEmbeds] = None
        self.negative_prompt_embeds: Optional[PromptEmbeds] = None
        self.neutral_prompt_embeds: Optional[PromptEmbeds] = None
        self.target_class_embeds: Optional[PromptEmbeds] = None
        self.anchor_class_embeds: Optional[PromptEmbeds] = None
        slider_raw = self.config.get("slider", {})
        if isinstance(slider_raw, list):
            self.sliders: list[ConceptSliderTrainerConfig] = [
                ConceptSliderTrainerConfig(**s) for s in slider_raw
            ]
        else:
            self.sliders: list[ConceptSliderTrainerConfig] = [
                ConceptSliderTrainerConfig(**slider_raw)
            ]

        # 兼容旧代码路径：保留 self.slider 指向第一个 slider
        self.slider: ConceptSliderTrainerConfig = self.sliders[0]

        # 每个 slider 的 prompt embeds 缓存
        self.slider_prompt_embeds: list[dict[str, Optional[PromptEmbeds]]] = []

    def hook_before_train_loop(self):
        # do this before calling parent as it unloads the text encoder if requested
        if self.is_caching_text_embeddings:
            # make sure model is on cpu for this part so we don't oom.
            self.sd.unet.to("cpu")

        # cache unconditional embeds (blank prompt)
        with torch.no_grad():
            self.slider_prompt_embeds = []
            for s in self.sliders:
                for target_class in s.target_classes:
                    positive_prompt_embeds = (
                        self.sd.encode_prompt([s.positive_prompt.replace("[target]", target_class)])
                        .to(self.device_torch, dtype=self.sd.torch_dtype)
                        .detach()
                    )

                    target_class_embeds = (
                        self.sd.encode_prompt([target_class])
                        .to(self.device_torch, dtype=self.sd.torch_dtype)
                        .detach()
                    )

                    neutral_prompt_embeds = (
                        self.sd.encode_prompt([s.neutral_prompt.replace("[target]", target_class) if s.neutral_prompt is not None else target_class])
                        .to(self.device_torch, dtype=self.sd.torch_dtype)
                        .detach()
                    )

                    negative_prompt_embeds = (
                        self.sd.encode_prompt([s.negative_prompt.replace("[target]", target_class)])
                        .to(self.device_torch, dtype=self.sd.torch_dtype)
                        .detach()
                    )

                    anchor_class_embeds: Optional[PromptEmbeds] = None
                    if s.anchor_class is not None:
                        anchor_class_embeds = (
                            self.sd.encode_prompt([s.anchor_class.replace("[target]", target_class)])
                            .to(self.device_torch, dtype=self.sd.torch_dtype)
                            .detach()
                        )

                    self.slider_prompt_embeds.append(
                        {
                            "positive": positive_prompt_embeds,
                            "target": target_class_embeds,
                            "negative": negative_prompt_embeds,
                            "neutral": neutral_prompt_embeds,
                            "anchor": anchor_class_embeds,
                        }
                    )

        # call parent
        super().hook_before_train_loop()

    def get_linearity_loss(
        self,
        noisy_latents,
        embeds,
        timesteps,
        batch,
        class_pred: torch.Tensor,
        m,
    ) -> torch.Tensor:
        was_unet_training = self.sd.unet.training
        old_m = self.network.multiplier

        if self.anchor_class_embeds is not None:
            # 把embeds和timestemps切成两半，因为外部传入的是concatenated的
            timesteps, _ = timesteps.chunk(2, dim=0)
            noisy_latents, _ = noisy_latents.chunk(2, dim=0)

        t = timesteps.to(device=class_pred.device, dtype=class_pred.dtype)
        linearity_weight = self.slider.linearity_weight * (t / 1000.0) ** 2
        with torch.no_grad():
            # do left pred
            self.sd.unet.eval()
            self.network.set_multiplier(m - self.slider.linearity_delta)
            pred = self.sd.predict_noise(
                latents=noisy_latents,
                conditional_embeddings=embeds,
                timestep=timesteps,
                guidance_scale=1.0,
                guidance_embedding_scale=1.0,
                batch=batch,
            )
            class_pred_left = pred

            # do right pred
            self.network.set_multiplier(m + self.slider.linearity_delta)
            pred = self.sd.predict_noise(
                latents=noisy_latents,
                conditional_embeddings=embeds,
                timestep=timesteps,
                guidance_scale=1.0,
                guidance_embedding_scale=1.0,
                batch=batch,
            )
            class_pred_right = pred

        # restore unet training
        if was_unet_training:
            self.sd.unet.train()
        # restore network multiplier
        self.network.set_multiplier(old_m)

        linearity_mid = (class_pred_left + class_pred_right) / 2.0

        if self.slider.linearity_loss_type == "mae":
            linearity_loss = (
                torch.nn.functional.l1_loss(class_pred, linearity_mid, reduction="none")
                .flatten(start_dim=1)
                .mean(dim=1)
            )
            linearity_loss = (linearity_loss * linearity_weight).mean()
        elif self.slider.linearity_loss_type == "mse":
            linearity_loss = (
                torch.nn.functional.mse_loss(
                    class_pred, linearity_mid, reduction="none"
                )
                .flatten(start_dim=1)
                .mean(dim=1)
            )
            linearity_loss = (linearity_loss * linearity_weight).mean()
        elif self.slider.linearity_loss_type == "smooth_l1":
            with torch.no_grad():
                abs_e = (class_pred.detach() - linearity_mid).abs().flatten().float()
                beta_target = torch.quantile(abs_e, q=self.slider.linearity_huber_beta).item()

            linearity_loss = (
                torch.nn.functional.smooth_l1_loss(
                    class_pred, linearity_mid, beta=float(beta_target), reduction="none"
                )
                .flatten(start_dim=1)
                .mean(dim=1)
            )
            linearity_loss = (linearity_loss * linearity_weight).mean()
        else:
            linearity_loss = torch.zeros((), device=class_pred.device, dtype=class_pred.dtype)
        return linearity_loss

    def color_cast_loss_gated(
        self,
        pred_a: torch.Tensor,
        pred_b: torch.Tensor,
        *,
        kernel_size: int = 3,
        weight: float = 1.0,
        loss: str = "huber",      # "huber" or "charbonnier" or "mse"
        huber_beta: float = 0.1,
        charbonnier_eps: float = 1e-3,
    ) -> torch.Tensor:
        """
        仅低频抑制的“色彩偏移”惩罚（pred 域）。
        - 先对 pred 差分做低通（avg pool 到 ~down_to）
        - 再取全局通道均值（B,C）
        - 去掉亮度公共分量，只惩罚通道间差异（chroma cast）
        - 用平滑损失聚合为标量

        pred_a/pred_b: (B,C,H,W)
        """
        if pred_a.shape != pred_b.shape:
            raise ValueError(f"Shape mismatch: {pred_a.shape} vs {pred_b.shape}")
        if pred_a.ndim != 4:
            raise ValueError(f"Expected (B,C,H,W), got {pred_a.shape}")

        d = pred_a - pred_b  # (B,C,H,W)

        d_lp = torch.nn.functional.avg_pool2d(d, kernel_size=kernel_size, stride=(kernel_size + 1) // 2)

        # --- chroma mean shift (remove luminance component) ---
        mean_shift = d_lp.mean(dim=(2, 3))                       # (B,C)
        chroma = mean_shift - mean_shift.mean(dim=1, keepdim=True)  # (B,C)

        # --- smooth penalty ---
        if loss == "mse":
            chroma_cast_per = (chroma ** 2).mean(dim=1)  # (B,)
            luma_cast_per = (mean_shift ** 2).mean(dim=1)  # (B,)
        elif loss == "charbonnier":
            chroma_cast_per = (chroma ** 2 + (charbonnier_eps ** 2)).sqrt().mean(dim=1)
            luma_cast_per = (mean_shift ** 2 + (charbonnier_eps ** 2)).sqrt().mean(dim=1)  # (B,)
        else:  # "huber"
            beta_target = torch.quantile(chroma.abs().float(), q=huber_beta).item()
            chroma_cast_per = torch.nn.functional.smooth_l1_loss(
                chroma, torch.zeros_like(chroma), beta=float(beta_target), reduction="none"
            ).mean(dim=1)
            beta_target = torch.quantile(mean_shift.abs().float(), q=huber_beta).item()
            luma_cast_per = torch.nn.functional.smooth_l1_loss(
                mean_shift, torch.zeros_like(mean_shift), beta=float(beta_target), reduction="none"
            ).mean(dim=1)

        return weight * (chroma_cast_per.mean() + luma_cast_per.mean())

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
        # todo for embeddings, we need to run without trigger words
        was_unet_training = self.sd.unet.training
        was_network_active = False
        if self.network is not None:
            was_network_active = self.network.is_active
            self.network.is_active = False

        num_sliders = max(1, len(self.sliders))
        slider_idx = torch.randint(0, num_sliders, (1,)).item()
        self.slider = self.sliders[slider_idx]
        self.positive_prompt_embeds = self.slider_prompt_embeds[slider_idx].get("positive")
        self.target_class_embeds = self.slider_prompt_embeds[slider_idx].get("target")
        self.negative_prompt_embeds = self.slider_prompt_embeds[slider_idx].get("negative")
        self.neutral_prompt_embeds = self.slider_prompt_embeds[slider_idx].get("neutral")
        self.anchor_class_embeds = self.slider_prompt_embeds[slider_idx].get("anchor")
        # do out prior preds first
        with torch.no_grad():
            dtype = get_torch_dtype(self.train_config.dtype)
            self.sd.unet.eval()
            noisy_latents = noisy_latents.to(self.device_torch, dtype=dtype).detach()

            batch_size = noisy_latents.shape[0]

            positive_embeds = concat_prompt_embeds(
                [self.positive_prompt_embeds] * batch_size
            ).to(self.device_torch, dtype=dtype)
            target_class_embeds = concat_prompt_embeds(
                [self.target_class_embeds] * batch_size
            ).to(self.device_torch, dtype=dtype)
            negative_embeds = concat_prompt_embeds(
                [self.negative_prompt_embeds] * batch_size
            ).to(self.device_torch, dtype=dtype)
            neutral_embeds = concat_prompt_embeds(
                [self.neutral_prompt_embeds] * batch_size
            ).to(self.device_torch, dtype=dtype)

            if self.anchor_class_embeds is not None:
                anchor_embeds = concat_prompt_embeds(
                    [self.anchor_class_embeds] * batch_size
                ).to(self.device_torch, dtype=dtype)

            if self.anchor_class_embeds is not None:
                # if we have an anchor, do it
                combo_embeds = concat_prompt_embeds(
                    [
                        positive_embeds,
                        neutral_embeds,
                        negative_embeds,
                        anchor_embeds,
                    ]
                )
                num_embeds = 4
            else:
                combo_embeds = concat_prompt_embeds(
                    [positive_embeds, neutral_embeds, negative_embeds]
                )
                num_embeds = 3

            # do them in one batch, VRAM should handle it since we are no grad
            combo_pred = self.sd.predict_noise(
                latents=torch.cat([noisy_latents] * num_embeds, dim=0),
                conditional_embeddings=combo_embeds,
                timestep=torch.cat([timesteps] * num_embeds, dim=0),
                guidance_scale=1.0,
                guidance_embedding_scale=1.0,
                batch=batch,
            )

            if self.anchor_class_embeds is not None:
                positive_pred, neutral_pred, negative_pred, anchor_target = (
                    combo_pred.chunk(4, dim=0)
                )
            else:
                anchor_target = None
                positive_pred, neutral_pred, negative_pred = combo_pred.chunk(3, dim=0)

            # calculate the targets
            guidance_scale = self.slider.guidance_strength

            # enhance_positive_target = neutral_pred + guidance_scale * (
            #     positive_pred - negative_pred
            # )
            # enhance_negative_target = neutral_pred + guidance_scale * (
            #     negative_pred - positive_pred
            # )
            # erase_negative_target = neutral_pred - guidance_scale * (
            #     negative_pred - positive_pred
            # )
            # erase_positive_target = neutral_pred - guidance_scale * (
            #     positive_pred - negative_pred
            # )

            positive = (positive_pred - neutral_pred) - (negative_pred - neutral_pred)
            negative = (negative_pred - neutral_pred) - (positive_pred - neutral_pred)

            enhance_positive_target = neutral_pred + guidance_scale * positive
            enhance_negative_target = neutral_pred + guidance_scale * negative
            erase_negative_target = neutral_pred - guidance_scale * negative
            erase_positive_target = neutral_pred - guidance_scale * positive

            # normalize to neutral std/mean
            enhance_positive_target = norm_like_tensor(
                enhance_positive_target, neutral_pred
            )
            enhance_negative_target = norm_like_tensor(
                enhance_negative_target, neutral_pred
            )
            erase_negative_target = norm_like_tensor(
                erase_negative_target, neutral_pred
            )
            erase_positive_target = norm_like_tensor(
                erase_positive_target, neutral_pred
            )

            if was_unet_training:
                self.sd.unet.train()

            # restore network
            if self.network is not None:
                self.network.is_active = was_network_active

            if self.anchor_class_embeds is not None:
                # do a grad inference with our target prompt
                embeds = concat_prompt_embeds([target_class_embeds, anchor_embeds]).to(
                    self.device_torch, dtype=dtype
                )

                noisy_latents = torch.cat([noisy_latents, noisy_latents], dim=0).to(
                    self.device_torch, dtype=dtype
                )
                timesteps = torch.cat([timesteps, timesteps], dim=0)
            else:
                embeds = target_class_embeds.to(self.device_torch, dtype=dtype)

        # do positive first
        m = self.slider.multiplier
        self.network.set_multiplier(m)
        pred = self.sd.predict_noise(
            latents=noisy_latents,
            conditional_embeddings=embeds,
            timestep=timesteps,
            guidance_scale=1.0,
            guidance_embedding_scale=1.0,
            batch=batch,
        )

        if self.anchor_class_embeds is not None:
            class_pred, anchor_pred = pred.chunk(2, dim=0)
        else:
            class_pred = pred
            anchor_pred = None

        # enhance positive loss
        enhance_loss = torch.nn.functional.mse_loss(class_pred, enhance_positive_target)

        # erase_loss = torch.nn.functional.mse_loss(class_pred, erase_negative_target)

        if anchor_target is None:
            anchor_loss = torch.zeros_like(enhance_loss)
        else:
            anchor_loss = torch.nn.functional.mse_loss(anchor_pred, anchor_target)

        anchor_loss = anchor_loss * self.slider.anchor_strength

        linearity_pos_loss = self.get_linearity_loss(noisy_latents, target_class_embeds, timesteps, batch, class_pred, m)

        cc_pos_loss = self.color_cast_loss_gated(class_pred, neutral_pred)

        # send backward now because gradient checkpointing needs network polarity intact
        total_pos_loss = (enhance_loss + anchor_loss) / 2.0 + linearity_pos_loss + cc_pos_loss
        total_pos_loss.backward()
        total_pos_loss = total_pos_loss.detach()

        # now do negative
        m = -self.slider.multiplier
        self.network.set_multiplier(m)
        pred = self.sd.predict_noise(
            latents=noisy_latents,
            conditional_embeddings=embeds,
            timestep=timesteps,
            guidance_scale=1.0,
            guidance_embedding_scale=1.0,
            batch=batch,
        )

        if self.anchor_class_embeds is not None:
            class_pred, anchor_pred = pred.chunk(2, dim=0)
        else:
            class_pred = pred
            anchor_pred = None

        # enhance negative loss
        # enhance_loss = torch.nn.functional.mse_loss(class_pred, enhance_negative_target)
        erase_loss = torch.nn.functional.mse_loss(class_pred, erase_positive_target)

        if anchor_target is None:
            anchor_loss = torch.zeros_like(erase_loss)
        else:
            anchor_loss = torch.nn.functional.mse_loss(anchor_pred, anchor_target)
        anchor_loss = anchor_loss * self.slider.anchor_strength

        linearity_neg_loss = self.get_linearity_loss(noisy_latents, target_class_embeds, timesteps, batch, class_pred, m)

        cc_neg_loss = self.color_cast_loss_gated(class_pred, neutral_pred)

        total_neg_loss = (erase_loss + anchor_loss) / 2.0 + linearity_neg_loss + cc_neg_loss
        total_neg_loss.backward()
        total_neg_loss = total_neg_loss.detach()

        self.network.set_multiplier(1.0)

        total_loss = (total_pos_loss + total_neg_loss) / 2.0

        # add a grad so backward works right
        total_loss.requires_grad_(True)
        return total_loss
