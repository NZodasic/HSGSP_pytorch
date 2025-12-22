import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import time
class ModelEvaluator:
    """Comprehensive model evaluation"""
   
    def __init__(self, config):
        self.config = config
   
    def evaluate_model(self,
                      model: torch.nn.Module,
                      dataloader: DataLoader,
                      dataset_name: str = "") -> Dict:
        """Model evaluation"""
        print(f"\nEvaluating model on {dataset_name}...")
        # Basic metrics
        loss, accuracy, top5_acc = self._compute_basic_metrics(model, dataloader)
       
        # Per-class metrics
        class_metrics = self._compute_per_class_metrics(model, dataloader)
       
        # Inference speed
        inference_time = self._measure_inference_speed(model, dataloader)
       
        # Model complexity
        complexity_metrics = self._compute_model_complexity(model)
       
        results = {
            'dataset': dataset_name,
            'loss': loss,
            'accuracy': accuracy,
            'top5_accuracy': top5_acc,
            'per_class_accuracy': class_metrics['per_class_acc'],
            'precision': class_metrics['precision'],
            'recall': class_metrics['recall'],
            'f1_score': class_metrics['f1'],
            'inference_time_ms': inference_time,
            'model_size_mb': complexity_metrics['size_mb'],
            'total_params': complexity_metrics['total_params'],
            'flops': complexity_metrics['flops']
        }
       
        return results
   
    def _compute_basic_metrics(self,
                              model: torch.nn.Module,
                              dataloader: DataLoader) -> Tuple[float, float, float]:
        """Compute loss, accuracy, top5 accuracy"""
        device = next(model.parameters()).device
        model.eval()
        total_loss = 0.0
        correct_1 = 0
        correct_5 = 0
        total = 0
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device) if y_batch.dim() == 1 else torch.argmax(y_batch, dim=1).to(device)
                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)
                total_loss += loss.item() * x_batch.size(0)
                _, pred_top1 = torch.topk(predictions, 1, dim=1)
                correct_1 += (pred_top1.view(-1) == y_batch).sum().item()
                _, pred_top5 = torch.topk(predictions, 5, dim=1)
                correct_5 += (pred_top5 == y_batch.unsqueeze(1)).any(dim=1).sum().item()
                total += x_batch.size(0)
        avg_loss = total_loss / max(total, 1)
        acc1 = correct_1 / max(total, 1)
        acc5 = correct_5 / max(total, 1)
        return avg_loss, acc1, acc5
   
    def _compute_per_class_metrics(self,
                                  model: torch.nn.Module,
                                  dataloader: DataLoader) -> Dict:
        """Compute per-class metrics"""
        device = next(model.parameters()).device
        model.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(device)
                predictions = model(x_batch)
                all_predictions.append(torch.argmax(predictions, dim=1).cpu().numpy())
                all_labels.append(y_batch.cpu().numpy() if y_batch.dim() == 1 else torch.argmax(y_batch, dim=1).cpu().numpy())
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
       
        # Compute metrics
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
       
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='macro'
        )
       
        # Per-class accuracy
        confusion = confusion_matrix(all_labels, all_predictions)
        per_class_acc = np.diag(confusion) / confusion.sum(axis=1)
       
        return {
            'per_class_acc': per_class_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': confusion
        }
   
    def _measure_inference_speed(self,
                                model: torch.nn.Module,
                                dataloader: DataLoader,
                                num_samples: int = 100) -> float:
        """Measure average inference time"""
        device = next(model.parameters()).device
        model.eval()
        times = []
        batches_to_take = max(1, int(np.ceil(num_samples / float(self.config.batch_size))))
        warmup_batches = min(5, batches_to_take)
        measured_batches = 0
        with torch.no_grad():
            for idx, (x_batch, _) in enumerate(dataloader):
                x_batch = x_batch.to(device)
                if idx < warmup_batches:
                    _ = model(x_batch)
                    continue
                start_time = time.perf_counter()
                _ = model(x_batch)
                end_time = time.perf_counter()
                batch_time_ms = (end_time - start_time) * 1000.0
                per_sample_ms = batch_time_ms / float(x_batch.size(0))
                times.append(per_sample_ms)
                measured_batches += 1
                if measured_batches >= batches_to_take:
                    break
        return float(np.mean(times)) if times else 0.0
   
    def _compute_model_complexity(self, model: torch.nn.Module) -> Dict:
        """Compute model complexity metrics"""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
       
        # Estimate model size
        size_mb = total_params * 4 / (1024 * 1024) # Assuming float32
       
        # Estimate FLOPs
        from models.model_utils import ModelUtils
        flops = ModelUtils.compute_flops(model, self.config.input_shape_cifar10)
       
        return {
            'total_params': total_params,
            'size_mb': size_mb,
            'flops': flops
        }
    def benchmark_inference(self,
                            model: torch.nn.Module,
                            batch_size: int = 1,
                            warmup_runs: int = 20,
                            measure_runs: int = 100,
                            dataloader: Optional[DataLoader] = None,
                            dtype: torch.dtype = torch.float32) -> Dict[str, float]:
        """High-resolution inference benchmark with optional dataloader input."""
        device = next(model.parameters()).device
        if dataloader is not None:
            dl_iter = iter(dataloader)
            try:
                sample_batch = next(dl_iter)[0]
            except (StopIteration, TypeError):
                raise ValueError("Dataloader must yield (inputs, labels) tuples with non-empty batches.")
            inputs = sample_batch[:batch_size].to(device=device, dtype=dtype)
        else:
            input_shape = self.config.input_shape_cifar10
            inputs = torch.randn((batch_size,) + input_shape, device=device, dtype=dtype)
        model.eval()
        with torch.no_grad():
            _ = model(inputs)
            for _ in range(max(0, warmup_runs)):
                _ = model(inputs)
            timings = []
            for _ in range(max(1, measure_runs)):
                start = time.perf_counter()
                _ = model(inputs)
                end = time.perf_counter()
                timings.append((end - start) * 1000.0)
        batch_latency_ms = float(np.mean(timings))
        per_sample_ms = batch_latency_ms / max(1, inputs.size(0))
        throughput = 1000.0 / per_sample_ms if per_sample_ms > 0 else float('inf')
        return {
            'batch_latency_ms': batch_latency_ms,
            'per_sample_latency_ms': per_sample_ms,
            'throughput_samples_per_sec': throughput,
            'batch_size': int(inputs.size(0))
        }
    def estimate_accuracy(self,
                          model: torch.nn.Module,
                          dataloader: DataLoader,
                          max_batches: Optional[int] = None) -> Dict[str, float]:
        """Estimate loss/accuracy on (optionally truncated) dataloader."""
        if max_batches is not None and max_batches > 0:
            dataloader = iter(dataloader)
            batches = [next(dataloader) for _ in range(max_batches)]
            dataloader = batches
        else:
            dataloader = list(dataloader)
        loss, acc, _ = self._compute_basic_metrics(model, dataloader)
        return {'loss': loss, 'accuracy': acc}
   
    def compare_models(self,
                      original_model: torch.nn.Module,
                      pruned_model: torch.nn.Module,
                      dataloader: DataLoader,
                      dataset_name: str = "") -> Dict:
        """Compare original and pruned models"""
        print(f"\nComparing models on {dataset_name}...")
       
        # Evaluate both models
        original_results = self.evaluate_model(original_model, dataloader, f"{dataset_name} (Original)")
        pruned_results = self.evaluate_model(pruned_model, dataloader, f"{dataset_name} (Pruned)")
       
        # Compute improvements
        comparison = {
            'accuracy_drop': original_results['accuracy'] - pruned_results['accuracy'],
            'speedup': original_results['inference_time_ms'] / pruned_results['inference_time_ms'],
            'compression_ratio': original_results['total_params'] / pruned_results['total_params'],
            'size_reduction': 1 - (pruned_results['model_size_mb'] / original_results['model_size_mb']),
            'flops_reduction': 1 - (pruned_results['flops'] / original_results['flops'])
        }
       
        return {
            'original': original_results,
            'pruned': pruned_results,
            'comparison': comparison
        }