import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.fft as sfft
from typing import Dict, Tuple, List, Optional
class FrequencyAnalyzer:
    """Frequency domain analysis for filters"""
   
    def __init__(self, config):
        self.config = config
        self.frequency_bands = config.frequency_bands
   
    def compute_dct_2d(self, filters: np.ndarray) -> np.ndarray:
        """Compute 2D DCT for filters"""
        # filters shape: (height, width, in_channels, out_channels)
        h, w, in_c, out_c = filters.shape
       
        # Reshape for batch processing
        filters_reshaped = filters.transpose(2, 3, 0, 1).reshape(in_c * out_c, h, w)
       
        # Compute 2D DCT
        dct_result = sfft.dctn(filters_reshaped, type=2, norm='ortho', axes=(1, 2))
       
        # Reshape back
        dct_result = dct_result.reshape(in_c, out_c, h, w).transpose(2, 3, 0, 1)
       
        return dct_result
   
    def extract_frequency_bands(self, dct_filters: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract frequency bands from DCT coefficients"""
        h, w = dct_filters.shape[:2]
        bands = {}
       
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            # Create frequency mask
            mask = self._create_frequency_mask(h, w, low_freq, high_freq)
           
            # Apply mask to extract band
            band_coeffs = dct_filters * mask[..., np.newaxis, np.newaxis]
            bands[band_name] = band_coeffs
       
        return bands
   
    def _create_frequency_mask(self, h: int, w: int,
                               low_freq: float, high_freq: float) -> np.ndarray:
        """Create a frequency band mask"""
        # Create meshgrid for frequencies
        u = np.arange(h)[:, np.newaxis] / (h - 1) if h > 1 else np.zeros((h, 1))
        v = np.arange(w)[np.newaxis, :] / (w - 1) if w > 1 else np.zeros((1, w))
       
        # Compute normalized frequency magnitude
        freq_magnitude = np.sqrt(u**2 + v**2)
       
        # Create band mask
        mask = ((freq_magnitude >= low_freq) & (freq_magnitude < high_freq)).astype(np.float32)
       
        return mask
   
    def compute_band_energy(self, band_coeffs: np.ndarray) -> np.ndarray:
        """Compute energy for each filter in a frequency band"""
        # Energy per filter (L2 norm of coefficients)
        energy = np.sqrt(np.sum(band_coeffs**2, axis=(0, 1, 2)))
        return energy
   
    def analyze_layer_frequency(self, layer: nn.Conv2d) -> Dict[str, np.ndarray]:
        """Analyze frequency characteristics of a convolutional layer"""
        # Get layer weights
        weights = layer.weight.detach().cpu().numpy().transpose(2, 3, 1, 0)  # to (h, w, in, out)
       
        # Compute DCT
        dct_filters = self.compute_dct_2d(weights)
       
        # Extract frequency bands
        bands = self.extract_frequency_bands(dct_filters)
       
        # Compute band energies
        band_energies = {}
        for band_name, band_coeffs in bands.items():
            band_energies[band_name] = self.compute_band_energy(band_coeffs)
       
        return band_energies
    def frequency_profile(self, model: nn.Module) -> Dict[str, Dict[str, float]]:
        """Compute frequency band energy profile for each Conv2d layer.
        This function does followings:
        - Extracts filters (kernels) from each Conv2d layer
        - Computes a 2D DCT-II (orthonormal) for each filter
        - Partitions coefficients into low/mid/high frequency bands
        - Computes L2 norm (energy) per band per layer
        Args:
            model: torch.nn.Module containing Conv2d layers
            dataset: Unused placeholder to match requested signature
        Returns:
            Dict[str, Dict[str, float]] mapping layer name -> {low, mid, high} energies
        """
        # Default band definitions aligned with Config.frequency_bands
        band_defs = {
            'low': (0.0, 0.25),
            'mid': (0.25, 0.5),
            'high': (0.5, 1.0)
        }
        def create_frequency_mask(h: int, w: int, low: float, high: float) -> np.ndarray:
            # Use DCT index-based "frequency" in [0,1], not FFT frequencies.
            # For DCT-II, indices k,l in [0..H-1],[0..W-1] increase spatial frequency.
            # Normalize indices so max radial is 1.
            u = (np.arange(h) / (h - 1)) if h > 1 else np.zeros(h)
            v = (np.arange(w) / (w - 1)) if w > 1 else np.zeros(w)
            U, V = np.meshgrid(u, v, indexing='ij')
            r = np.sqrt(U**2 + V**2) / np.sqrt(2.0)
            return ((r >= low) & (r < high)).astype(np.float32)
        def dct2_ortho(x: np.ndarray) -> np.ndarray:
            # Apply DCT-II along spatial dimensions (H, W) with orthonormal scaling
            # x shape: (H, W, Cin, Cout)
            # DCT along H (axis 0)
            y_h = sfft.dct(x, axis=0, type=2, norm='ortho')
            # DCT along W (axis 1)
            y2 = sfft.dct(y_h, axis=1, type=2, norm='ortho')
            return y2
        frequency_maps: Dict[str, Dict[str, float]] = {}
        for name, module in model.named_modules():
            # Only process Conv2d modules with weight
            if not isinstance(module, nn.Conv2d):
                continue
            kernel = module.weight.detach().cpu().numpy().transpose(2, 3, 1, 0)  # to (H, W, Cin, Cout)
            h, w, _, _ = kernel.shape
            dct_filters = dct2_ortho(kernel)  # (H, W, Cin, Cout)
            # Compute energies per band
            energies = {}
            for band_name, (lo, hi) in band_defs.items():
                mask = create_frequency_mask(h, w, lo, hi)  # (H, W)
                mask = mask[..., np.newaxis, np.newaxis]  # broadcast over Cin, Cout
                band_coeffs = dct_filters * mask
                # L2 norm of all band coefficients (scalar)
                energy = float(np.linalg.norm(band_coeffs))
                energies[band_name] = energy
            frequency_maps[name] = energies
        return frequency_maps
    def _compute_activation_statistics(
        self,
        model: nn.Module,
        dataset: Optional[DataLoader],
        max_batches: int = 200,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute per-filter activation statistics (mean |activation| and std) for Conv2d layers.
        Returns normalized (z-scored) statistics per layer.
        """
        if dataset is None:
            return {}
        conv_layers = ModelUtils.get_conv_layers(model)
        if not conv_layers:
            return {}
        device = next(model.parameters()).device
        abs_acc: Dict[str, np.ndarray] = {}
        sq_acc: Dict[str, np.ndarray] = {}
        count_acc: Dict[str, float] = {}
        for layer in conv_layers:
            cout = layer.out_channels
            abs_acc[layer._get_name()] = np.zeros((cout,), np.float64)
            sq_acc[layer._get_name()] = np.zeros((cout,), np.float64)
            count_acc[layer._get_name()] = 0.0
        batches = 0
        model.eval()
        with torch.no_grad():
            for i, (xb, _) in enumerate(dataset):
                if batches >= max_batches:
                    break
                xb = xb.to(device)
                activations = []
                hooks = []
                def hook_fn(m, i, o):
                    activations.append(o)
                for l in conv_layers:
                    hooks.append(l.register_forward_hook(hook_fn))
                _ = model(xb)
                for h in hooks:
                    h.remove()
                for l_idx, act in enumerate(activations):
                    layer_name = conv_layers[l_idx]._get_name()
                    act_np = act.detach().cpu().numpy()
                    abs_acc[layer_name] += np.sum(np.abs(act_np), axis=(0, 2, 3)).astype(np.float64)
                    sq_acc[layer_name] += np.sum(np.square(act_np), axis=(0, 2, 3)).astype(np.float64)
                    count_acc[layer_name] += float(act_np.shape[0] * act_np.shape[2] * act_np.shape[3])
                batches += 1
        if batches == 0:
            return {}
        def _normalize(arr: np.ndarray) -> np.ndarray:
            mu = float(np.mean(arr))
            sigma = float(np.std(arr))
            if sigma < 1e-12:
                return np.zeros_like(arr, dtype=np.float64)
            return (arr - mu) / (sigma + 1e-12)
        stats: Dict[str, Dict[str, np.ndarray]] = {}
        for layer in conv_layers:
            layer_name = layer._get_name()
            if layer_name not in abs_acc:
                continue
            total_count = count_acc.get(layer_name, 0.0)
            if total_count <= 0.0:
                continue
            mean_abs = abs_acc[layer_name] / total_count
            mean_sq = sq_acc[layer_name] / total_count
            variance = np.maximum(mean_sq - np.square(mean_abs), 0.0)
            std = np.sqrt(variance)
            stats[layer_name] = {
                "mean_abs": _normalize(mean_abs),
                "std": _normalize(std),
            }
        return stats
    def compute_frequency_importance_scores(
        self,
        model: nn.Module,
        band_weights: Dict[str, float] | None = None,
        energy_exponent: float = 0.5,
        normalize_per_layer: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Compute per-filter importance scores for each Conv2d layer using DCT-based
        frequency band energies.
        For each Conv2d kernel K [H,W,Cin,Cout]:
          1) Compute 2D DCT-II (orthonormal) over spatial dims (H,W).
          2) Partition coefficients into low/mid/high bands (DCT index radius).
          3) For each output channel j, compute band energies E_b[j] = ||coeffs_b||_2.
          4) Let T[j] = sum_b E_b[j], r_b[j] = E_b[j] / (T[j] + eps).
          5) Importance[j] = (sum_b w_b * r_b[j]) * (T[j] ** energy_exponent).
        - band_weights (default): {'low': 0.6, 'mid': 0.3, 'high': 0.1}
        - energy_exponent: soft boost for overall energy magnitude; 0 disables.
        - normalize_per_layer: min-max normalize importance to [0,1] within a layer.
        Returns:
            Dict[layer_name, importance_scores (np.ndarray of shape [Cout])]
        """
        # Defaults
        if band_weights is None:
            band_weights = {'low': 0.6, 'mid': 0.3, 'high': 0.1}
        def dct2_ortho(x: np.ndarray) -> np.ndarray:
            # x: (H, W, Cin, Cout)
            # DCT along H (axis=0)
            y_h = sfft.dct(x, axis=0, type=2, norm='ortho')
            # DCT along W (axis=1)
            y2 = sfft.dct(y_h, axis=1, type=2, norm='ortho')
            return y2
        def create_mask(h: int, w: int, low: float, high: float) -> np.ndarray:
            u = (np.arange(h) / (h - 1)) if h > 1 else np.zeros(h)
            v = (np.arange(w) / (w - 1)) if w > 1 else np.zeros(w)
            U, V = np.meshgrid(u, v, indexing='ij')
            r = np.sqrt(U**2 + V**2) / np.sqrt(2.0)
            return ((r >= low) & (r < high)).astype(np.float32)
        # Band definitions aligned with config defaults
        band_defs = self.config.frequency_bands if hasattr(self.config, 'frequency_bands') and self.config.frequency_bands else {
            'low': (0.0, 0.25),
            'mid': (0.25, 0.5),
            'high': (0.5, 1.0),
        }
        importance: Dict[str, np.ndarray] = {}
        for name, module in model.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            kernel = module.weight.detach().cpu().numpy().transpose(2, 3, 1, 0)  # to (H, W, Cin, Cout)
            h, w, _, cout = kernel.shape
            dct_k = dct2_ortho(kernel)  # (H, W, Cin, Cout)
            band_energy = {}
            total_energy = np.zeros((cout,), dtype=np.float64)
            for bname, (lo, hi) in band_defs.items():
                mask = create_mask(h, w, lo, hi)
                mask = mask[..., np.newaxis, np.newaxis]  # (H, W, 1, 1)
                coeffs_b = dct_k * mask
                # Energy per output channel
                e_b = np.sqrt(np.sum(np.square(coeffs_b), axis=(0, 1, 2)))  # (Cout,)
                band_energy[bname] = e_b
                total_energy += e_b
            eps = 1e-12
            # Weighted ratio component
            weighted_ratio = np.zeros((cout,), dtype=np.float64)
            for bname, e_b in band_energy.items():
                w_b = float(band_weights.get(bname, 0.0)) if band_weights is not None else 0.0
                weighted_ratio += w_b * (e_b / (total_energy + eps))
            # Magnitude boost
            mag = np.power(np.maximum(total_energy, 0.0) + eps, float(energy_exponent))
            scores = weighted_ratio * mag
            if normalize_per_layer:
                s_min = float(np.min(scores))
                s_max = float(np.max(scores))
                rng = s_max - s_min
                if rng < 1e-12:
                    scores_norm = np.zeros_like(scores, dtype=np.float64)
                else:
                    scores_norm = (scores - s_min) / (rng + 1e-12)
                importance[name] = scores_norm.astype(np.float32)
            else:
                importance[name] = scores.astype(np.float32)
        return importance
    def build_frequency_relevance_net(hidden_units: int = 32) -> nn.Module:
        """
        PyTorch version of FrequencyRelevanceNet.
        Input: 3-dim vector [low, mid, high] (ratios or raw energies)
        Output: 3-dim softmax weights over the three bands
        """
        class FrequencyRelevanceNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(3, hidden_units)
                self.fc2 = nn.Linear(hidden_units, 3)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                return F.softmax(self.fc2(x), dim=-1)
        
        return FrequencyRelevanceNet()
   
    def compute_frequency_importance_scores_with_frn(
        self,
        model: nn.Module,
        frn_model: Optional[nn.Module] = None,
        use_ratios_as_input: bool = True,
        energy_exponent: float = 0.5,
        normalize_per_layer: bool = True,
        fallback_band_weights: Optional[Dict[str, float]] = None,
        activation_dataset: Optional[DataLoader] = None,
        activation_max_batches: int = 200,
    ) -> Dict[str, np.ndarray]:
        """
        Compute per-filter importance scores for each Conv2d layer using DCT-based
        frequency band energies and an optional PyTorch FrequencyRelevanceNet.
        For each Conv2d kernel K [H,W,Cin,Cout]:
        1) Compute DCT-II (orthonormal) over (H,W).
        2) Split coefficients into low/mid/high bands via index-radius masks.
        3) Per output channel j, compute band energies E_b[j] (Frobenius over H,W,Cin).
        4) Let T[j] = sum_b E_b[j]; ratios r_b[j] = E_b[j] / (T[j] + eps).
        5) Build FRN input x_j = [r_low, r_mid, r_high] if use_ratios_as_input else [E_low, E_mid, E_high] (log1p).
        6) If frn_model provided: w_j = softmax(FRN(x_j)); else use fallback_band_weights.
        7) score_j = (sum_b w_b[j] * r_b[j]) * (T[j] ** energy_exponent).
        8) Optionally min–max normalize scores within the layer.
        Returns:
            Dict[layer_name, scores_per_filter] with shape [Cout] per layer.
        """
        # ---------- defaults ----------
        if fallback_band_weights is None:
            fallback_band_weights = {'low': 0.6, 'mid': 0.3, 'high': 0.1}
        activation_stats = self._compute_activation_statistics(
            model=model,
            dataset=activation_dataset,
            max_batches=activation_max_batches,
        )
        def dct2_ortho(x: np.ndarray) -> np.ndarray:
            # x: (H, W, Cin, Cout)
            # DCT along H (axis=0)
            y_h = sfft.dct(x, axis=0, type=2, norm='ortho')
            # DCT along W (axis=1)
            y2 = sfft.dct(y_h, axis=1, type=2, norm='ortho')
            return y2
        def create_mask(h: int, w: int, low: float, high: float) -> np.ndarray:
            u = (np.arange(h) / (h - 1)) if h > 1 else np.zeros(h)
            v = (np.arange(w) / (w - 1)) if w > 1 else np.zeros(w)
            U, V = np.meshgrid(u, v, indexing='ij')
            r = np.sqrt(U**2 + V**2) / np.sqrt(2.0)
            return ((r >= low) & (r < high)).astype(np.float32)
        band_defs = getattr(self.config, "frequency_bands", None) or {
            'low': (0.0, 0.25),
            'mid': (0.25, 0.5),
            'high': (0.5, 1.0),
        }
        importance: Dict[str, np.ndarray] = {}
        device = next(frn_model.parameters()).device if frn_model else torch.device('cpu')
        # ---------- iterate Conv2d modules ----------
        for name, module in model.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            kernel = module.weight.detach().cpu().numpy().transpose(2, 3, 1, 0)  # to (H, W, Cin, Cout)
            h, w, _, cout = kernel.shape
            dct_k = dct2_ortho(kernel)  # (H, W, Cin, Cout)
            band_energy = {}
            total_energy = np.zeros((cout,), dtype=np.float64)
            # band energies per output channel
            for bname, (lo, hi) in band_defs.items():
                mask = create_mask(h, w, lo, hi)
                mask = mask[..., np.newaxis, np.newaxis]  # (H, W, 1, 1)
                coeffs_b = dct_k * mask
                e_b = np.sqrt(np.sum(np.square(coeffs_b), axis=(0, 1, 2)))  # (Cout,)
                band_energy[bname] = e_b
                total_energy += e_b
            eps = 1e-12
            # ratios r_b[j] = E_b[j] / T[j]
            ratios = {b: e / (total_energy + eps) for b, e in band_energy.items()}
            if use_ratios_as_input:
                feature_core = np.stack(
                    [ratios['low'], ratios['mid'], ratios['high']],
                    axis=1,
                )
            else:
                feature_core = np.stack(
                    [band_energy['low'], band_energy['mid'], band_energy['high']],
                    axis=1,
                )
                feature_core = np.log1p(feature_core)
            expected_dim = None
            if frn_model is not None:
                expected_dim = frn_model.fc1.in_features
            layer_activation_stats = activation_stats.get(name) if activation_stats else None
            extra_features: Optional[np.ndarray] = None
            if layer_activation_stats is not None:
                extra_features = np.stack(
                    [
                        layer_activation_stats['mean_abs'],
                        layer_activation_stats['std'],
                    ],
                    axis=1,
                )
            if expected_dim is not None:
                needed_extra = max(expected_dim - feature_core.shape[1], 0)
                if extra_features is None:
                    if needed_extra > 0:
                        extra_features = np.zeros((cout, needed_extra), dtype=np.float64)
                else:
                    if extra_features.shape[1] < needed_extra:
                        pad = np.zeros((cout, needed_extra - extra_features.shape[1]), dtype=np.float64)
                        extra_features = np.concatenate([extra_features, pad], axis=1)
                    elif extra_features.shape[1] > needed_extra:
                        extra_features = extra_features[:, :needed_extra]
            else:
                if extra_features is not None:
                    expected_dim = feature_core.shape[1] + extra_features.shape[1]
            if extra_features is not None and extra_features.shape[1] > 0:
                x_mat = np.concatenate([feature_core, extra_features], axis=1)
            else:
                x_mat = feature_core
            # Compute per-filter band weights
            if frn_model is not None:
                # PyTorch forward pass (no grad)
                x_torch = torch.from_numpy(x_mat.astype(np.float32)).to(device)
                with torch.no_grad():
                    w_torch = frn_model(x_torch)  # (Cout,3), softmax
                w_np = w_torch.cpu().numpy().astype(np.float64)
                w_low, w_mid, w_high = w_np[:, 0], w_np[:, 1], w_np[:, 2]
            else:
                # Fallback static weights
                w_low = np.full((cout,), float(fallback_band_weights.get('low', 0.0)), dtype=np.float64)
                w_mid = np.full((cout,), float(fallback_band_weights.get('mid', 0.0)), dtype=np.float64)
                w_high = np.full((cout,), float(fallback_band_weights.get('high', 0.0)), dtype=np.float64)
            # Weighted mixture over band *ratios*
            weighted_ratio = (
                w_low * ratios['low'] +
                w_mid * ratios['mid'] +
                w_high * ratios['high']
            ) # (Cout,)
            # Magnitude boost T^alpha
            mag = np.power(np.maximum(total_energy, 0.0) + eps, float(energy_exponent)) # (Cout,)
            scores = weighted_ratio * mag # (Cout,)
            # Optional per-layer min–max normalization
            s_min = float(np.min(scores))
            s_max = float(np.max(scores))
            rng = s_max - s_min
            scores = np.zeros_like(scores, dtype=np.float64) if rng < 1e-12 else (scores - s_min) / (rng + 1e-12)
            importance[name] = scores.astype(np.float32)
        return importance