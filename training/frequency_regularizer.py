import copy
import torch
from torch import nn
import torch.nn.functional as F
import torch.fft
from typing import Dict, Iterable, Tuple, Union, Optional
def _dct2(x: torch.Tensor) -> torch.Tensor:
    """Apply orthonormal 2-D DCT (type-II) over the last two axes."""
    # x: [..., H, W]
    # DCT along H (dim=-2)
    x = torch.fft.rfft(x, dim=-2, norm="ortho")
    # Scale for type-II
    x[..., 0, :] *= torch.sqrt(torch.tensor(1.0 / 2.0, device=x.device))
    # DCT along W (dim=-1)
    x = torch.fft.rfft(x, dim=-1, norm="ortho")
    x[..., :, 0] *= torch.sqrt(torch.tensor(1.0 / 2.0, device=x.device))
    return x.real
def _reshape_spatial(features: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """Flatten batch/channel dims so DCT runs per (sample, channel)."""
    features = features.float()
    shape = features.shape
    height = shape[1]
    width = shape[2]
    return features.reshape(-1, height, width), height, width
def compute_spectral_entropy(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute the mean spectral entropy of 4-D activations.
    Args:
        features: Tensor shaped [B, H, W, C].
        eps: Numerical stability constant.
    Returns:
        Scalar tensor with the average entropy across B*C channels.
    """
    flat, height, width = _reshape_spatial(features)
    coeffs = _dct2(flat)
    energy = coeffs ** 2
    total = torch.sum(energy, dim=[1, 2], keepdim=True) + eps
    probs = energy / total
    entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=eps)), dim=[1, 2])
    return torch.mean(entropy)
def frequency_entropy_loss(
    features: torch.Tensor,
    target_entropy: Union[float, torch.Tensor],
    beta: float = 0.01,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Penalize deviation from a desired spectral entropy.
    Args:
        features: Activations [B, H, W, C].
        target_entropy: Desired entropy level (scalar).
        beta: Strength of the penalty term.
        eps: Numeric stability constant.
    Returns:
        Tuple of (scalar loss, current entropy).
    """
    entropy = compute_spectral_entropy(features, eps=eps)
    target = torch.tensor(target_entropy, dtype=torch.float32, device=features.device)
    loss = beta * (entropy - target) ** 2
    return loss, entropy
class SpectralEntropyRegularizer:
    """
    Helper that measures layer-wise spectral entropy and returns a weighted loss.
    """
    def __init__(
        self,
        model: nn.Module,
        layer_names: Iterable[str],
        target_entropies: Dict[str, float],
        beta: float = 0.01,
        layer_weights: Optional[Dict[str, float]] = None,
    ):
        self.layer_names = list(layer_names)
        self.beta = float(beta)
        self.layer_weights = layer_weights or {}
        self._targets = torch.tensor([float(target_entropies.get(name, 0.0)) for name in self.layer_names])
        self.hooks = []
    def register_hooks(self, model: nn.Module):
        activations = {}
        def hook_fn(name):
            def hook(m, i, o):
                activations[name] = o.detach()
            return hook
        for name in self.layer_names:
            module = dict(model.named_modules())[name]
            self.hooks.append(module.register_forward_hook(hook_fn(name)))
        return activations
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
    def __call__(self, model: nn.Module, inputs: torch.Tensor):
        activations = self.register_hooks(model)
        _ = model(inputs)
        self.remove_hooks()
        feats = [activations.get(name) for name in self.layer_names]
        total_loss = torch.zeros((), dtype=torch.float32, device=inputs.device)
        entropy_map: Dict[str, torch.Tensor] = {}
        for idx, (layer_name, feat) in enumerate(zip(self.layer_names, feats)):
            if feat is None:
                continue
            layer_scale = float(self.layer_weights.get(layer_name, 1.0))
            layer_beta = layer_scale * self.beta
            layer_loss, entropy = frequency_entropy_loss(
                feat,
                target_entropy=self._targets[idx],
                beta=layer_beta,
            )
            total_loss += layer_loss
            entropy_map[layer_name] = entropy
        return total_loss, entropy_map
class FrequencyRegularizedModel(nn.Module):
    """
    Wraps an existing model and injects spectral-entropy loss into train_step.
    """
    def __init__(
        self,
        base_model: nn.Module,
        spectral_regularizer: SpectralEntropyRegularizer,
    ):
        super().__init__()
        self._base_model = base_model
        self._spectral_regularizer = spectral_regularizer
    def forward(self, inputs):
        return self._base_model(inputs)
    # In PyTorch, training loop is manual, so provide a method to compute extra loss
    def compute_reg_loss(self, inputs):
        return self._spectral_regularizer(self._base_model, inputs)[0]