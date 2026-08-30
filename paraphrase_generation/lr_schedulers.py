import math


def _validate_learning_rate(name, value):
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite, non-negative number")


class ConstantLearningRate:
    """Constant learning-rate baseline."""

    requires_metric = False

    def __init__(self, initial_lr):
        _validate_learning_rate("initial_lr", initial_lr)
        self.initial_lr = initial_lr

    def lr(self):
        return self.initial_lr

    def step(self, metric=None):
        if metric is not None:
            raise ValueError("ConstantLearningRate does not accept a metric")


class StepDecay:
    """Multiply the learning rate by gamma every step_size optimizer updates."""

    requires_metric = False

    def __init__(self, initial_lr, step_size, gamma=0.5, min_lr=0.0):
        _validate_learning_rate("initial_lr", initial_lr)
        _validate_learning_rate("min_lr", min_lr)
        if step_size < 1:
            raise ValueError("step_size must be at least 1")
        if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in the interval (0, 1]")
        if min_lr > initial_lr:
            raise ValueError("min_lr cannot exceed initial_lr")

        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr
        self.step_count = 0

    def lr(self):
        decay_count = self.step_count // self.step_size
        return max(self.min_lr, self.initial_lr * self.gamma ** decay_count)

    def step(self, metric=None):
        if metric is not None:
            raise ValueError("StepDecay does not accept a metric")
        self.step_count += 1


class CosineDecay:
    """Cosine decay from initial_lr to min_lr over total_steps updates."""

    requires_metric = False

    def __init__(self, initial_lr, total_steps, min_lr=0.0):
        _validate_learning_rate("initial_lr", initial_lr)
        _validate_learning_rate("min_lr", min_lr)
        if total_steps < 1:
            raise ValueError("total_steps must be at least 1")
        if min_lr > initial_lr:
            raise ValueError("min_lr cannot exceed initial_lr")

        self.initial_lr = initial_lr
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.step_count = 0

    def lr(self):
        progress = min(self.step_count, self.total_steps) / self.total_steps
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.initial_lr - self.min_lr) * cosine_factor

    def step(self, metric=None):
        if metric is not None:
            raise ValueError("CosineDecay does not accept a metric")
        self.step_count += 1


class LinearDecay:
    """Linear decay from initial_lr to min_lr over total_steps updates."""

    requires_metric = False

    def __init__(self, initial_lr, total_steps, min_lr=0.0):
        _validate_learning_rate("initial_lr", initial_lr)
        _validate_learning_rate("min_lr", min_lr)
        if total_steps < 1:
            raise ValueError("total_steps must be at least 1")
        if min_lr > initial_lr:
            raise ValueError("min_lr cannot exceed initial_lr")

        self.initial_lr = initial_lr
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.step_count = 0

    def lr(self):
        progress = min(self.step_count, self.total_steps) / self.total_steps
        return max(
            self.min_lr,
            self.initial_lr - (self.initial_lr - self.min_lr) * progress,
        )

    def step(self, metric=None):
        if metric is not None:
            raise ValueError("LinearDecay does not accept a metric")
        self.step_count += 1


class InverseSquareRoot:
    """Optional linear warmup followed by inverse-square-root decay."""

    requires_metric = False

    def __init__(self, initial_lr, warmup_steps=0, min_lr=0.0):
        _validate_learning_rate("initial_lr", initial_lr)
        _validate_learning_rate("min_lr", min_lr)
        if warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        if min_lr > initial_lr:
            raise ValueError("min_lr cannot exceed initial_lr")

        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.step_count = 0

    def lr(self):
        update_number = self.step_count + 1
        if self.warmup_steps > 0:
            scale = min(
                update_number / self.warmup_steps,
                math.sqrt(self.warmup_steps / update_number),
            )
        else:
            scale = 1.0 / math.sqrt(update_number)
        return max(self.min_lr, self.initial_lr * scale)

    def step(self, metric=None):
        if metric is not None:
            raise ValueError("InverseSquareRoot does not accept a metric")
        self.step_count += 1


class MetricDependent:
    """Reduce the learning rate when a maximized metric stops improving."""

    requires_metric = True

    def __init__(
        self,
        initial_lr,
        factor=0.5,
        patience=1,
        min_lr=0.0,
        threshold=1e-8,
    ):
        _validate_learning_rate("initial_lr", initial_lr)
        _validate_learning_rate("min_lr", min_lr)
        if not math.isfinite(factor) or not 0.0 < factor < 1.0:
            raise ValueError("factor must be in the interval (0, 1)")
        if patience < 0:
            raise ValueError("patience cannot be negative")
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("threshold must be finite and non-negative")
        if min_lr > initial_lr:
            raise ValueError("min_lr cannot exceed initial_lr")

        self.current_lr = initial_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        self.best_metric = None
        self.bad_epochs = 0

    def lr(self):
        return self.current_lr

    def step(self, metric=None):
        if metric is None or not math.isfinite(metric):
            raise ValueError("MetricDependent requires a finite metric")

        if self.best_metric is None or metric > self.best_metric + self.threshold:
            self.best_metric = metric
            self.bad_epochs = 0
            return

        self.bad_epochs += 1
        if self.bad_epochs > self.patience:
            self.current_lr = max(self.min_lr, self.current_lr * self.factor)
            self.bad_epochs = 0
