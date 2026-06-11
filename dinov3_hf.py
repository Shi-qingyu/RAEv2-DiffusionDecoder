# --------------------------------------------------------
# HuggingFace DINOv3 encoder wrapper for RAE-style targets.
# Implements the DINOv3 feature extraction path used in model_raev2.py.
# --------------------------------------------------------

import os
import re
from math import sqrt
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel


DINOV3_IMAGE_MEAN = (0.485, 0.456, 0.406)
DINOV3_IMAGE_STD = (0.229, 0.224, 0.225)

DINOv3_HF_MODEL_IDS = {
    "s16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "s16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "b16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "l16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "h16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
}


def _infer_base_config(model_ref: str) -> Optional[str]:
    name = os.path.basename(str(model_ref)).lower()
    patterns = (
        ("vit7b16", "7b16"),
        ("vith16plus", "h16plus"),
        ("vithplus", "h16plus"),
        ("vitl16", "l16"),
        ("vitb16", "b16"),
        ("vits16plus", "s16plus"),
        ("vitsplus", "s16plus"),
        ("vits16", "s16"),
    )
    for pattern, base in patterns:
        if pattern in name:
            return base
    return None


def _parse_dinov3_config(model_config: str) -> Tuple[str, List[str], str]:
    """Return (base_config, flags, model_ref_without_flags)."""
    cfg = str(model_config)
    bracket_match = re.match(r"^(.+?)\[([^\]]+)\]$", cfg)
    if bracket_match:
        model_ref = bracket_match.group(1)
        flags = [flag.strip() for flag in bracket_match.group(2).split(",")]
        base_config = _infer_base_config(model_ref) or model_ref
        return base_config, flags, model_ref

    known_bases = sorted(DINOv3_HF_MODEL_IDS, key=len, reverse=True)
    for base in known_bases:
        if cfg.startswith(base):
            suffix = cfg[len(base):]
            if not suffix:
                return base, [], base
            flags = []
            remaining = suffix
            while remaining:
                if remaining.startswith("norm"):
                    flags.append("norm")
                    remaining = remaining[len("norm"):]
                else:
                    return _infer_base_config(cfg) or cfg, [], cfg
            return base, flags, base

    return _infer_base_config(cfg) or cfg, [], cfg


def _as_pair_size(value) -> int:
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


class DINOv3HFEncoder(nn.Module):
    """HuggingFace DINOv3 ViT encoder with model_raev2-compatible outputs.

    Public `forward()`/`forward_features()` expect image tensors in [0, 1],
    then resize and apply ImageNet mean/std before calling the HF model.
    If the input is already normalized pixel_values, pass is_normalized=True.

    Config examples:
        "l16"
        "l16[norm]"
        "l16[layers=11.13.15.17.19.21.23]"
        "facebook/dinov3-vitl16-pretrain-lvd1689m"
    """

    def __init__(
        self,
        dinov3_path: Optional[str] = None,
        model_config: str = "l16",
        normalize: bool = True,
        resolution: int = 256,
        local_files_only: bool = False,
        **from_pretrained_kwargs,
    ):
        super().__init__()
        self.model_config = model_config
        self.base_config, self.flags, model_ref = _parse_dinov3_config(model_config)
        self.resolution = resolution

        if dinov3_path is None:
            dinov3_path = DINOv3_HF_MODEL_IDS.get(self.base_config, model_ref)

        load_kwargs = dict(from_pretrained_kwargs)
        load_kwargs.setdefault("local_files_only", local_files_only)
        self.encoder = AutoModel.from_pretrained(dinov3_path, **load_kwargs)
        self.encoder.requires_grad_(False)
        self.encoder.eval()

        self.model = self.encoder
        self.hidden_size = int(self.encoder.config.hidden_size)
        self.embed_dim = self.hidden_size
        self.patch_size = _as_pair_size(getattr(self.encoder.config, "patch_size", 16))
        self.num_register_tokens = int(getattr(self.encoder.config, "num_register_tokens", 0))

        mean = torch.tensor(DINOV3_IMAGE_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(DINOV3_IMAGE_STD).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

        if normalize and "norm" not in self.flags:
            self._strip_final_norm_affine()

    @property
    def prefix_tokens(self) -> int:
        return 1 + self.num_register_tokens

    def _strip_final_norm_affine(self) -> None:
        norm = self._final_norm_module()
        if norm is None:
            return
        eps = getattr(norm, "eps", 1e-5)
        new_norm = nn.LayerNorm(self.hidden_size, eps=eps, elementwise_affine=False)
        try:
            param = next(self.encoder.parameters())
            new_norm = new_norm.to(device=param.device, dtype=param.dtype)
        except StopIteration:
            pass

        if hasattr(self.encoder, "norm"):
            self.encoder.norm = new_norm
        elif hasattr(self.encoder, "layernorm"):
            self.encoder.layernorm = new_norm

    def _final_norm_module(self) -> Optional[nn.Module]:
        if hasattr(self.encoder, "norm"):
            return self.encoder.norm
        if hasattr(self.encoder, "layernorm"):
            return self.encoder.layernorm
        return None

    def _apply_final_norm(self, hidden_state: torch.Tensor) -> torch.Tensor:
        norm = self._final_norm_module()
        return norm(hidden_state) if norm is not None else hidden_state

    def _normalize_unit_images(self, x: torch.Tensor) -> torch.Tensor:
        """Resize and ImageNet-normalize images in [0, 1]."""
        if x.shape[-2:] != (self.resolution, self.resolution):
            x = F.interpolate(
                x,
                size=(self.resolution, self.resolution),
                mode="bicubic",
                align_corners=False,
            )
        return (x - self.image_mean.to(x.device, x.dtype)) / self.image_std.to(x.device, x.dtype)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess raw images in [0, 255], matching model_raev2.py."""
        return self._normalize_unit_images(x / 255.0)

    def _reshape_patch_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        bsz, num_tokens, channels = patch_tokens.shape
        spatial = int(sqrt(num_tokens))
        assert spatial * spatial == num_tokens, (
            f"Cannot reshape {num_tokens} DINOv3 tokens into a square feature map."
        )
        return patch_tokens.transpose(1, 2).reshape(bsz, channels, spatial, spatial)

    def _format_features(
        self,
        features: Dict[str, Optional[torch.Tensor]],
        return_dict: bool,
        reshape_to_2d: bool,
    ):
        if return_dict:
            return features
        patch_tokens = features["x_norm_patchtokens"]
        return self._reshape_patch_tokens(patch_tokens) if reshape_to_2d else patch_tokens

    def _forward_features_from_pixel_values(
        self,
        x: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        out = self.encoder(pixel_values=x, return_dict=True)
        sequence = out.last_hidden_state
        return {
            "x_norm_clstoken": sequence[:, 0, :],
            "x_norm_patchtokens": sequence[:, self.prefix_tokens:, :],
        }

    def forward_features(
        self,
        x: torch.Tensor,
        *,
        is_normalized: bool = False,
        return_dict: bool = False,
        reshape_to_2d: bool = True,
    ):
        pixel_values = x if is_normalized else self._normalize_unit_images(x)
        features = self._forward_features_from_pixel_values(pixel_values)
        return self._format_features(features, return_dict, reshape_to_2d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x, return_dict=False, reshape_to_2d=True)


class DINOv3HFMultiLayerSimpleAddEncoder(DINOv3HFEncoder):
    """Average normalized patch tokens from selected DINOv3 layers.

    This mirrors `DINOv3MultiLayerSimpleAddEncoder` in model_raev2.py:
    1. get selected intermediate layers
    2. keep patch tokens only
    3. average selected layers
    4. add the mean patch token from the final selected layer
    """

    DEFAULT_LAYERS = {
        "l16": [11, 13, 15, 17, 19, 21, 23],
    }

    def __init__(
        self,
        dinov3_path: Optional[str] = None,
        model_config: str = "l16",
        layer_indices: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        super().__init__(dinov3_path=dinov3_path, model_config=model_config, **kwargs)
        if layer_indices is not None:
            self.layer_indices = [int(idx) for idx in layer_indices]
        else:
            layers_flag = [flag for flag in self.flags if flag.startswith("layers=")]
            if layers_flag:
                self.layer_indices = [
                    int(idx) for idx in layers_flag[0].split("=", 1)[1].split(".")
                ]
            else:
                self.layer_indices = self.DEFAULT_LAYERS.get(self.base_config, [2, 5, 8, 11])

    def _hidden_state_for_layer(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        layer_idx: int,
    ) -> torch.Tensor:
        if layer_idx < 0:
            return hidden_states[layer_idx]

        num_layers = int(getattr(self.encoder.config, "num_hidden_layers", 0))
        if num_layers and len(hidden_states) == num_layers + 1:
            return hidden_states[layer_idx + 1]
        if num_layers and len(hidden_states) == num_layers:
            return hidden_states[layer_idx]
        if len(hidden_states) > layer_idx + 1:
            return hidden_states[layer_idx + 1]
        return hidden_states[layer_idx]

    def _forward_features_from_pixel_values(
        self,
        x: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        out = self.encoder(pixel_values=x, output_hidden_states=True, return_dict=True)
        if out.hidden_states is None:
            raise RuntimeError("DINOv3 HF model did not return hidden_states.")

        outputs = []
        for layer_idx in self.layer_indices:
            hidden_state = self._hidden_state_for_layer(out.hidden_states, layer_idx)
            hidden_state = self._apply_final_norm(hidden_state)
            outputs.append(hidden_state[:, self.prefix_tokens:, :])

        patch_tokens = torch.stack(outputs, dim=0).mean(dim=0)
        final_mean = outputs[-1].mean(dim=1, keepdim=True)
        patch_tokens = patch_tokens + final_mean
        return {
            "x_norm_clstoken": final_mean.squeeze(1),
            "x_norm_patchtokens": patch_tokens,
        }


DINOv3Encoder = DINOv3HFEncoder
DINOv3MultiLayerSimpleAddEncoder = DINOv3HFMultiLayerSimpleAddEncoder
