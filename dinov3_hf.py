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
from transformers import AutoModel, AutoImageProcessor


DINOV3_IMAGE_MEAN = (0.485, 0.456, 0.406)
DINOV3_IMAGE_STD = (0.229, 0.224, 0.225)


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
        dino_resolution: int = 256,
    ):
        super().__init__()
        self.resolution = dino_resolution

        if dinov3_path is None:
            dinov3_path = "facebook/dinov3-vitl16-pretrain-lvd1689m"

        self.encoder = AutoModel.from_pretrained(dinov3_path)
        self.processor = AutoImageProcessor.from_pretrained(dinov3_path)
        self.encoder.requires_grad_(False)
        self.encoder.eval()

        self.model = self.encoder
        self.hidden_size = int(self.encoder.config.hidden_size)
        self.embed_dim = self.hidden_size
        self.patch_size = _as_pair_size(getattr(self.encoder.config, "patch_size", 16))
        self.num_register_tokens = int(getattr(self.encoder.config, "num_register_tokens", 0))

        mean = torch.tensor(self.processor.image_mean).view(1, 3, 1, 1)
        std = torch.tensor(self.processor.image_std).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

    @property
    def prefix_tokens(self) -> int:
        return 1 + self.num_register_tokens

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

    def reshape_to_2d(
        self,
        features: Dict[str, Optional[torch.Tensor]],
    ):
        patch_tokens = features["x_norm_patchtokens"]
        bsz, num_tokens, channels = patch_tokens.shape
        spatial = int(sqrt(num_tokens))
        assert spatial * spatial == num_tokens, (
            f"Cannot reshape {num_tokens} DINOv3 tokens into a square feature map."
        )
        return patch_tokens.transpose(1, 2).reshape(bsz, channels, spatial, spatial)

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
        images: torch.Tensor,
    ):
        pixel_values = self._normalize_unit_images(images)
        features = self._forward_features_from_pixel_values(pixel_values)
        return self.reshape_to_2d(features)

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
        dinov3_path: Optional[str],
        layer_indices: Optional[Sequence[int]],
        **kwargs,
    ):
        super().__init__(dinov3_path=dinov3_path, **kwargs)
        self.layer_indices = [int(idx) for idx in layer_indices]

    def _forward_features_from_pixel_values(
        self,
        x: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        out = self.encoder(pixel_values=x, output_hidden_states=True, return_dict=True)
        if out.hidden_states is None:
            raise RuntimeError("DINOv3 HF model did not return hidden_states.")

        outputs = []
        for layer_idx in self.layer_indices:
            hidden_state = out.hidden_states[layer_idx + 1]
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
