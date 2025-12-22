import torch
from torch import nn
import numpy as np
from typing import List, Dict, Tuple, Optional

class ModelUtils:
    """Utilities for model manipulation and analysis"""
   
    @staticmethod
    def get_conv_layers(model: nn.Module) -> List[nn.Conv2d]:
        """Extract all Conv2d modules from model"""
        conv_layers = []
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(module)
        return conv_layers
   
    @staticmethod
    def get_layer_weights(layer: nn.Module) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Get weights and biases from a layer"""
        weight = layer.weight.detach().cpu().numpy() if hasattr(layer, 'weight') else None
        bias = layer.bias.detach().cpu().numpy() if hasattr(layer, 'bias') and layer.bias is not None else None
        return weight, bias
   
    @staticmethod
    def set_layer_weights(layer: nn.Module,
                         weights: np.ndarray,
                         bias: Optional[np.ndarray] = None):
        """Set weights and biases for a layer"""
        if hasattr(layer, 'weight'):
            layer.weight.data = torch.from_numpy(weights).to(layer.weight.device).to(dtype=layer.weight.dtype)
        if bias is not None and hasattr(layer, 'bias'):
            layer.bias.data = torch.from_numpy(bias).to(layer.bias.device).to(dtype=layer.bias.dtype)
   
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count parameters in model"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
       
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params
        }
   
    @staticmethod
    def compute_flops(model: nn.Module, input_shape: Tuple[int, int, int]) -> int:
        """Estimate FLOPs for the model using a dummy input of given shape (channels, height, width)"""
        total_flops = 0
       
        def hook_fn(module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor):
            nonlocal total_flops
            if isinstance(module, nn.Conv2d):
                _, c_in, h_in, w_in = input[0].shape
                _, c_out, h_out, w_out = output.shape
                k_h, k_w = module.kernel_size
                groups = module.groups
                flops = 2 * h_out * w_out * k_h * k_w * (c_in // groups) * c_out
                if module.bias is not None:
                    flops += h_out * w_out * c_out
                total_flops += int(flops)
               
            elif isinstance(module, nn.Linear):
                in_dim = input[0].shape[-1]
                out_dim = module.out_features
                flops = 2 * in_dim * out_dim
                if module.bias is not None:
                    flops += out_dim
                total_flops += int(flops)
       
        hooks = [module.register_forward_hook(hook_fn) for module in model.modules()
                 if isinstance(module, (nn.Conv2d, nn.Linear))]
       
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        dummy_input = torch.randn(1, *input_shape, device=device, dtype=dtype)
        model.eval()
        with torch.no_grad():
            model(dummy_input)
       
        for hook in hooks:
            hook.remove()
       
        return total_flops