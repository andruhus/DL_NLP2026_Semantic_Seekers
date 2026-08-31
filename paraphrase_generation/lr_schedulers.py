import math

from typing_extensions import override


class ConstantLearningRate:
    """Constant learning-rate baseline."""

    def __init__(self, initial_lr, min_lr=0.0):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self._validate_learning_rate()

    def lr(self):
        return self.initial_lr

    def step(self):
        pass

    def _validate_learning_rate(self):
        if not math.isfinite(self.initial_lr) or self.initial_lr < 0.0:
            raise ValueError("initial_lr must be a finite, non-negative number")
        if not math.isfinite(self.min_lr) or self.min_lr < 0.0:
            raise ValueError("min_lr must be a finite, non-negative number")
        if self.min_lr > self.initial_lr:
            raise ValueError("min_lr cannot exceed initial_lr")


class StepDecay(ConstantLearningRate):
    """Multiply the learning rate by gamma every step_size optimizer updates."""

    def __init__(self, initial_lr, step_size, gamma=0.5, min_lr=0.0):
        self.step_size = step_size
        self.gamma = gamma
        self.step_count = 0
        super().__init__(initial_lr, min_lr)

    @override
    def lr(self):
        decay_count = self.step_count // self.step_size
        return max(self.min_lr, self.initial_lr * self.gamma**decay_count)

    @override
    def step(self):
        self.step_count += 1

    @override
    def _validate_learning_rate(self):
        if not math.isfinite(self.min_lr) or self.min_lr < 0.0:
            raise ValueError("min_lr must be a finite, non-negative number")
        if self.step_size < 1:
            raise ValueError("step_size must be at least 1")
        if not math.isfinite(self.gamma) or not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in the interval (0, 1]")
        return super()._validate_learning_rate()


class CosineDecay(ConstantLearningRate):
    """Cosine decay from initial_lr to min_lr over total_steps updates."""

    def __init__(self, initial_lr, total_steps, min_lr=0.0):
        self.total_steps = total_steps
        self.step_count = 0
        super().__init__(initial_lr, min_lr)

    @override
    def lr(self):
        progress = min(self.step_count, self.total_steps) / self.total_steps
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.initial_lr - self.min_lr) * cosine_factor

    @override
    def step(self, metric=None):
        if metric is not None:
            raise ValueError("CosineDecay does not accept a metric")
        self.step_count += 1

    @override
    def _validate_learning_rate(self):
        if self.total_steps < 1:
            raise ValueError("total_steps must be at least 1")
        return super()._validate_learning_rate()


class LinearDecay(ConstantLearningRate):
    """Linear decay from initial_lr to min_lr over total_steps updates."""

    def __init__(self, initial_lr, total_steps, min_lr=0.0):
        self.total_steps = total_steps
        self.step_count = 0
        super().__init__(initial_lr, min_lr)

    @override
    def lr(self):
        progress = min(self.step_count, self.total_steps) / self.total_steps
        return max(
            self.min_lr,
            self.initial_lr - (self.initial_lr - self.min_lr) * progress,
        )

    @override
    def step(self):
        self.step_count += 1

    @override
    def _validate_learning_rate(self):
        if self.total_steps < 1:
            raise ValueError("total_steps must be at least 1")
        return super()._validate_learning_rate()


class InverseSquareRoot(ConstantLearningRate):
    """Optional linear warmup followed by inverse-square-root decay."""

    def __init__(self, initial_lr, warmup_steps=0, min_lr=0.0):
        self.warmup_steps = warmup_steps
        self.step_count = 0
        super().__init__(initial_lr, min_lr)

    @override
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

    @override
    def step(self):
        self.step_count += 1

    @override
    def _validate_learning_rate(self):
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        return super()._validate_learning_rate()


class MetricDependent(ConstantLearningRate):
    """Reduce the learning rate when a maximized metric stops improving."""

    requires_metric = True

    def __init__(self, initial_lr, factor=0.5, patience=1, min_lr=0.0, threshold=1e-8,):
        self.current_lr = initial_lr
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.best_metric = None
        self.bad_epochs = 0
        super().__init__(initial_lr, min_lr)

    @override
    def lr(self):
        return self.current_lr

    @override
    def step(self, metric):
        if not math.isfinite(metric):
            raise ValueError("MetricDependent requires a finite metric")

        if self.best_metric is None or metric > self.best_metric + self.threshold:
            self.best_metric = metric
            self.bad_epochs = 0
            return

        self.bad_epochs += 1
        if self.bad_epochs > self.patience:
            self.current_lr = max(self.min_lr, self.current_lr * self.factor)
            self.bad_epochs = 0

    @override
    def _validate_learning_rate(self):
        if not math.isfinite(self.factor) or not 0.0 < self.factor < 1.0:
            raise ValueError("factor must be in the interval (0, 1)")
        if self.patience < 0:
            raise ValueError("patience cannot be negative")
        if not math.isfinite(self.threshold) or self.threshold < 0.0:
            raise ValueError("threshold must be finite and non-negative")
        return super()._validate_learning_rate()
