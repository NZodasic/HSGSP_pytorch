import copy
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional
class Distiller(nn.Module):
    """
    Knowledge Distillation wrapper for PyTorch models.
    Trains a student to match ground-truth labels while also matching a
    softened distribution from a (frozen) teacher model.
    Total loss = alpha * student_loss(y_true, y_pred_student)
               + (1 - alpha) * (T^2) * KL(softmax_t(teacher) || softmax_t(student))
    Where softmax_t(z) applies temperature scaling.
    """
    def __init__(self,
                 student: nn.Module,
                 teacher: nn.Module,
                 alpha: float = 0.5,
                 temperature: float = 4.0,
                 name: Optional[str] = None):
        super().__init__()
        self.student = student
        self.teacher = teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.student_loss_fn = None
        self.distillation_loss_fn = None
        self.student_from_logits = False
        self.teacher_from_logits = False
    def compile(self,
                optimizer: torch.optim.Optimizer,
                metrics: Optional[List] = None,
                student_loss_fn: Optional[nn.Module] = None,
                distillation_loss_fn: Optional[nn.Module] = None,
                alpha: Optional[float] = None,
                temperature: Optional[float] = None,
                student_from_logits: Optional[bool] = None,
                teacher_from_logits: Optional[bool] = None,
                **kwargs):
        self.optimizer = optimizer
        self.metrics = metrics or []
        if alpha is not None:
            self.alpha = float(alpha)
        if temperature is not None:
            self.temperature = float(temperature)
        self.student_loss_fn = student_loss_fn or nn.CrossEntropyLoss()
        self.distillation_loss_fn = distillation_loss_fn or nn.KLDivLoss(reduction='batchmean')
        if student_from_logits is None:
            student_from_logits = self._infer_from_logits(self.student)
        if teacher_from_logits is None:
            teacher_from_logits = self._infer_from_logits(self.teacher)
        self.student_from_logits = bool(student_from_logits)
        self.teacher_from_logits = bool(teacher_from_logits)
    @staticmethod
    def _infer_from_logits(model: nn.Module) -> bool:
        if model is None:
            return False
        # In PyTorch, most models output logits unless explicitly softened
        # Check if last module is Softmax
        last_module = list(model.modules())[-1]
        if isinstance(last_module, nn.Softmax):
            return False
        return True
    def _soften(self, outputs: torch.Tensor, temperature: float, from_logits: bool) -> torch.Tensor:
        """Apply temperature scaling to logits or probabilities."""
        if from_logits:
            return F.softmax(outputs / temperature, dim=-1)
        eps = 1e-7
        probs = torch.clamp(outputs, min=eps, max=1.0)
        log_probs = torch.log(probs)
        return F.softmax(log_probs / temperature, dim=-1)
    def train_step(self, data):
        x, y_true = data
        self.optimizer.zero_grad()
        # Forward pass through student and teacher
        y_student = self.student(x)
        with torch.no_grad():
            y_teacher = self.teacher(x)
        # Student supervised loss (with labels)
        student_loss = self.student_loss_fn(y_student, y_true)
        # Distillation loss with temperature
        T = self.temperature
        p_teacher_t = self._soften(y_teacher, T, self.teacher_from_logits)
        p_student_t = self._soften(y_student, T, self.student_from_logits)
        distill_loss = self.distillation_loss_fn(torch.log(p_student_t), p_teacher_t)
        distill_loss *= (T * T) # standard KD scaling
        reg_loss = torch.tensor(0.0, device=student_loss.device)
        total_loss = self.alpha * student_loss + (1.0 - self.alpha) * distill_loss + reg_loss
        total_loss.backward()
        self.optimizer.step()
        # Update metrics if any
        metrics = {}
        for m in self.metrics:
            m.update_state(y_true, y_student)
            metrics[m.__class__.__name__] = m.result()
        # Log individual losses for monitoring
        metrics.update({
            'loss': total_loss.item(),
            'student_loss': student_loss.item(),
            'distillation_loss': distill_loss.item(),
            'reg_loss': reg_loss.item(),
        })
        return metrics
    def test_step(self, data):
        x, y_true = data
        y_student = self.student(x)
        with torch.no_grad():
            y_teacher = self.teacher(x)
        student_loss = self.student_loss_fn(y_student, y_true)
        T = self.temperature
        p_teacher_t = self._soften(y_teacher, T, self.teacher_from_logits)
        p_student_t = self._soften(y_student, T, self.student_from_logits)
        distill_loss = self.distillation_loss_fn(torch.log(p_student_t), p_teacher_t) * (T * T)
        reg_loss = torch.tensor(0.0, device=student_loss.device)
        total_loss = self.alpha * student_loss + (1.0 - self.alpha) * distill_loss + reg_loss
        metrics = {}
        for m in self.metrics:
            m.update_state(y_true, y_student)
            metrics[m.__class__.__name__] = m.result()
        metrics.update({
            'loss': total_loss.item(),
            'student_loss': student_loss.item(),
            'distillation_loss': distill_loss.item(),
            'reg_loss': reg_loss.item(),
        })
        return metrics
    # Expose student forward to behave like a plain model when needed
    def forward(self, inputs):
        return self.student(inputs)