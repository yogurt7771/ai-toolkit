from collections import OrderedDict
from typing import Optional
from copy import deepcopy

import torch
from tqdm import tqdm

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
        self.target_class: str = kwargs.get("target_class", "")
        self.target_classes: list[str] = kwargs.get("target_classes", [self.target_class])
        self.target_attributes: list[str] = kwargs.get("target_attributes", ["[target]"])
        self.anchor_class: Optional[str] = kwargs.get("anchor_class", None)
        self.multiplier: float = float(kwargs.get("multiplier", 1.0))
        # --- auxiliary losses (keep defaults consistent with function defaults) ---
        # cap aux losses
        self.cap_ratio: float = float(kwargs.get("cap_ratio", 0.2))
        self.cap_main_floor: float = float(kwargs.get("cap_main_floor", 1e-4))
        self.cap_eps: float = float(kwargs.get("cap_eps", 1e-5))
        # linearity loss
        self.linearity_weight: float = float(kwargs.get("linearity_weight", 1.0))
        self.linearity_delta: float = float(kwargs.get("linearity_delta", 0.1))
        self.linearity_loss_type: str = kwargs.get("linearity_loss_type", "smooth_l1")
        self.linearity_gate_threshold: float = float(kwargs.get("linearity_gate_threshold", 0.01))
        self.linearity_beta: float = float(kwargs.get("linearity_beta", 0.05))
        # color cast loss
        self.color_cast_weight: float = float(kwargs.get("color_cast_weight", 1.0))
        self.color_cast_loss_type: str = kwargs.get("color_cast_loss_type", "smooth_l1")
        self.color_cast_beta: float = float(kwargs.get("color_cast_beta", 0.05))
        self.color_cast_gate_threshold: float = float(kwargs.get("color_cast_gate_threshold", 0.01))
        self.color_cast_down_to: int = int(kwargs.get("color_cast_down_to", 8))


def norm_like_tensor(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Normalize the tensor to have the same mean and std as the target tensor."""
    tensor_mean = tensor.mean(dim=(1, 2, 3), keepdim=True)
    tensor_std = tensor.std(dim=(1, 2, 3), keepdim=True)
    target_mean = target.mean(dim=(1, 2, 3), keepdim=True)
    target_std = target.std(dim=(1, 2, 3), keepdim=True)
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

        # flatten the sliders
        slider_global: dict = self.config.get("slider", {})
        self.sliders: list[ConceptSliderTrainerConfig] = []
        for s_update in slider_global.get("prompts", []):
            s = deepcopy(slider_global)
            s.update(s_update)
            s = ConceptSliderTrainerConfig(**s)
            for target_class in s.target_classes:
                for target_attribute in s.target_attributes:
                    target = target_attribute.replace("[target]", target_class)
                    new_slider = deepcopy(s)
                    new_slider.target_class = target
                    new_slider.positive_prompt = new_slider.positive_prompt.replace(
                        "[target]", target
                    )
                    new_slider.neutral_prompt = (
                        new_slider.neutral_prompt.replace("[target]", target)
                        if new_slider.neutral_prompt is not None
                        else target
                    )
                    new_slider.negative_prompt = new_slider.negative_prompt.replace(
                        "[target]", target
                    )
                    new_slider.anchor_class = (
                        new_slider.anchor_class.replace("[target]", target)
                        if new_slider.anchor_class is not None
                        else None
                    )
                    self.sliders.append(new_slider)
        print(f"total sliders: {len(self.sliders)}")

        # 每个 slider 的 prompt embeds 缓存
        self.slider_prompt_embeds: list[dict[str, Optional[PromptEmbeds]]] = []

    def hook_before_train_loop(self):
        # do this before calling parent as it unloads the text encoder if requested
        if self.is_caching_text_embeddings:
            # make sure model is on cpu for this part so we don't oom.
            self.sd.unet.to("cpu")

        self.slider_prompt_embeds = []
        # cache unconditional embeds (blank prompt)
        with torch.no_grad():
            for new_slider in tqdm(self.sliders, desc="Encoding prompts"):
                positive_prompt_embeds = (
                    self.sd.encode_prompt([new_slider.positive_prompt])
                    .detach().to("cpu")
                )

                target_class_embeds = (
                    self.sd.encode_prompt([new_slider.target_class])
                    .detach().to("cpu")
                )

                neutral_prompt_embeds = (
                    self.sd.encode_prompt([new_slider.neutral_prompt])
                    .detach().to("cpu")
                )

                negative_prompt_embeds = (
                    self.sd.encode_prompt([new_slider.negative_prompt])
                    .detach().to("cpu")
                )

                anchor_class_embeds: Optional[PromptEmbeds] = None
                if new_slider.anchor_class is not None:
                    anchor_class_embeds = (
                        self.sd.encode_prompt([new_slider.anchor_class])
                        .detach().to("cpu")
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

    def cap_aux(
        self,
        aux: torch.Tensor,
        main: torch.Tensor,
        ratio: float = 0.2,
        main_floor: float = 1e-4,
        eps: float = 1e-5,
    ):
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

    def get_linearity_loss(
        self,
        noisy_latents,
        embeds,
        timesteps,
        batch,
        class_pred: torch.Tensor,
        m,
        *,
        weight: float = 1.0,
        delta: float = 0.1,
        loss_type: str = "smooth_l1",
        gate_threshold: float = 0.01,
        beta: float = 0.05,
    ) -> torch.Tensor:
        was_unet_training = self.sd.unet.training
        old_m = self.network.multiplier

        if self.anchor_class_embeds is not None:
            # 把embeds和timestemps切成两半，因为外部传入的是concatenated的
            timesteps, _ = timesteps.chunk(2, dim=0)
            noisy_latents, _ = noisy_latents.chunk(2, dim=0)

        t = timesteps.to(device=class_pred.device, dtype=class_pred.dtype)
        linearity_weight = weight * (t / 1000.0) ** 2
        with torch.no_grad():
            # do left pred
            self.sd.unet.eval()
            self.network.set_multiplier(m - delta)
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
            self.network.set_multiplier(m + delta)
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

        linearity_mid = (class_pred_left + class_pred_right)
        pred = class_pred * 2

        if loss_type == "mae":
            linearity_loss = (
                torch.nn.functional.l1_loss(pred, linearity_mid, reduction="none")
                .flatten(start_dim=1)
                .mean(dim=1)
            )
        elif loss_type == "mse":
            linearity_loss = (
                torch.nn.functional.mse_loss(
                    pred, linearity_mid, reduction="none"
                )
                .flatten(start_dim=1)
                .mean(dim=1)
            )
        elif loss_type == "smooth_l1":
            diff = pred - linearity_mid
            linearity_loss = self.smooth_l1_tensor_beta(diff, beta).flatten(start_dim=1).mean(dim=1)
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")
        thr = torch.as_tensor(gate_threshold, device=linearity_loss.device, dtype=linearity_loss.dtype)
        linearity_loss = torch.nn.functional.relu(linearity_loss - thr)
        return (linearity_loss * linearity_weight).mean()

    def color_cast_loss(
        self,
        pred: torch.Tensor,
        neutral: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        weight: float = 1.0,
        loss_type: str = "smooth_l1",  # "smooth_l1" or "mse"
        beta: float = 0.05,
        gate_threshold: float = 0.01,
        down_to: int = 8,
    ) -> torch.Tensor:
        """
        仅低频抑制的“色彩偏移”惩罚（pred 域）。
        - 先对 pred 差分做低通（avg pool 到 ~down_to）
        - 再取全局通道均值（B,C）
        - 去掉亮度公共分量，只惩罚通道间差异（chroma cast）
        - 用平滑损失聚合为标量

        pred_a/pred_b: (B,C,H,W)
        """
        def lowpass_to(d: torch.Tensor, down_to: int = 8):
            B, C, H, W = d.shape
            kH = max(1, H // max(1, down_to))
            kW = max(1, W // max(1, down_to))
            k = min(kH, kW)
            return torch.nn.functional.avg_pool2d(d, kernel_size=k, stride=k) if k > 1 else d

        if self.anchor_class_embeds is not None:
            # 把embeds和timestemps切成两半，因为外部传入的是concatenated的
            timesteps, _ = timesteps.chunk(2, dim=0)

        t = timesteps.to(device=pred.device, dtype=pred.dtype)
        t_norm = (t / 1000.0).clamp(0, 1)
        # 中间大两边小
        cc_weight = weight * (1.0 - (2.0 * t_norm - 1.0).abs()) ** 2

        d = pred - neutral  # (B,C,H,W)

        d_lp = lowpass_to(d, down_to=down_to)

        # --- chroma mean shift (remove luminance component) ---
        mean_shift = d_lp.mean(dim=(2, 3))
        luminance_shift = mean_shift.mean(dim=1, keepdim=True)
        chroma_shift = mean_shift - luminance_shift

        if loss_type == "mae":
            chroma_cast_per = (chroma_shift.abs()).mean(dim=1)
            luma_cast_per = (luminance_shift.abs()).mean(dim=1)
        elif loss_type == "mse":
            chroma_cast_per = (chroma_shift ** 2).mean(dim=1)
            luma_cast_per = (luminance_shift ** 2).mean(dim=1)
        elif loss_type == "smooth_l1":
            chroma_cast_per = self.smooth_l1_tensor_beta(chroma_shift, beta).mean(dim=1)
            luma_cast_per = self.smooth_l1_tensor_beta(luminance_shift, beta).mean(dim=1)
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")

        thr = torch.as_tensor(gate_threshold, device=chroma_cast_per.device, dtype=chroma_cast_per.dtype)
        chroma_cast_per = torch.nn.functional.relu(chroma_cast_per - thr)
        luma_cast_per = torch.nn.functional.relu(luma_cast_per - thr)
        return ((chroma_cast_per + luma_cast_per) * cc_weight).mean()

    def get_direct_loss(
        self,
        noisy_latents: torch.Tensor,
        embeds: PromptEmbeds,
        target_class_embeds: PromptEmbeds,
        neutral_pred: torch.Tensor,
        timesteps: torch.Tensor,
        batch: "DataLoaderBatchDTO",
        anchor_target: Optional[torch.Tensor],
        target: torch.Tensor,
        m: float,
    ):
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

        main_loss = torch.nn.functional.mse_loss(class_pred, target)

        if anchor_target is None:
            anchor_loss = torch.zeros_like(main_loss)
        else:
            anchor_loss = torch.nn.functional.mse_loss(anchor_pred, anchor_target)

        anchor_loss = anchor_loss * self.slider.anchor_strength

        linearity_loss = self.get_linearity_loss(
            noisy_latents,
            target_class_embeds,
            timesteps,
            batch,
            class_pred,
            m,
            weight=self.slider.linearity_weight,
            delta=self.slider.linearity_delta,
            loss_type=self.slider.linearity_loss_type,
            gate_threshold=self.slider.linearity_gate_threshold,
            beta=self.slider.linearity_beta,
        )

        cc_loss = self.color_cast_loss(
            class_pred,
            neutral_pred,
            timesteps,
            weight=self.slider.color_cast_weight,
            loss_type=self.slider.color_cast_loss_type,
            beta=self.slider.color_cast_beta,
            gate_threshold=self.slider.color_cast_gate_threshold,
            down_to=self.slider.color_cast_down_to,
        )

        anchor_loss = self.cap_aux(
            anchor_loss,
            main_loss,
            ratio=self.slider.cap_ratio,
            main_floor=self.slider.cap_main_floor,
            eps=self.slider.cap_eps,
        )
        cc_loss = self.cap_aux(
            cc_loss,
            main_loss,
            ratio=self.slider.cap_ratio,
            main_floor=self.slider.cap_main_floor,
            eps=self.slider.cap_eps,
        )
        linearity_loss = self.cap_aux(
            linearity_loss,
            main_loss,
            ratio=self.slider.cap_ratio,
            main_floor=self.slider.cap_main_floor,
            eps=self.slider.cap_eps,
        )
        total_loss = main_loss + anchor_loss + linearity_loss + cc_loss
        total_loss.backward()
        total_loss = total_loss.detach()

        return total_loss

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
        self.positive_prompt_embeds = self.slider_prompt_embeds[slider_idx].get("positive").to(self.device_torch, dtype=self.sd.torch_dtype)
        self.target_class_embeds = self.slider_prompt_embeds[slider_idx].get("target").to(self.device_torch, dtype=self.sd.torch_dtype)
        self.negative_prompt_embeds = self.slider_prompt_embeds[slider_idx].get("negative").to(self.device_torch, dtype=self.sd.torch_dtype)
        self.neutral_prompt_embeds = self.slider_prompt_embeds[slider_idx].get("neutral").to(self.device_torch, dtype=self.sd.torch_dtype)
        self.anchor_class_embeds = self.slider_prompt_embeds[slider_idx].get("anchor").to(self.device_torch, dtype=self.sd.torch_dtype)
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
            # negative = (negative_pred - neutral_pred) - (positive_pred - neutral_pred)

            enhance_positive_target = neutral_pred + guidance_scale * positive
            # enhance_negative_target = neutral_pred + guidance_scale * negative
            # erase_negative_target = neutral_pred - guidance_scale * negative
            erase_positive_target = neutral_pred - guidance_scale * positive

            # normalize to neutral std/mean
            enhance_positive_target = norm_like_tensor(
                enhance_positive_target, neutral_pred
            )
            # enhance_negative_target = norm_like_tensor(
            #     enhance_negative_target, neutral_pred
            # )
            # erase_negative_target = norm_like_tensor(
            #     erase_negative_target, neutral_pred
            # )
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
        total_pos_loss = self.get_direct_loss(
            noisy_latents,
            embeds,
            target_class_embeds,
            neutral_pred,
            timesteps,
            batch,
            anchor_target,
            enhance_positive_target,
            self.slider.multiplier,
        )

        # now do negative
        total_neg_loss = self.get_direct_loss(
            noisy_latents,
            embeds,
            target_class_embeds,
            neutral_pred,
            timesteps,
            batch,
            anchor_target,
            erase_positive_target,
            -self.slider.multiplier,
        )

        self.network.set_multiplier(1.0)

        total_loss = (total_pos_loss + total_neg_loss) / 2.0

        # add a grad so backward works right
        total_loss.requires_grad_(True)

        # unload the prompt embeds
        self.positive_prompt_embeds.to("cpu")
        self.target_class_embeds.to("cpu")
        self.negative_prompt_embeds.to("cpu")
        self.neutral_prompt_embeds.to("cpu")
        self.anchor_class_embeds.to("cpu")

        return total_loss
