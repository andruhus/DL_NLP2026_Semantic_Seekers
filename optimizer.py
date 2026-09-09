import math
from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        lr_sched=None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1])
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        if lr_sched is not None:
            scheduler_lr = getattr(lr_sched, "lr", None)
            if not callable(scheduler_lr):
                raise TypeError("lr_sched must provide a callable lr() method")
            initial_scheduled_lr = float(scheduler_lr())
            if not math.isfinite(initial_scheduled_lr) or initial_scheduled_lr < 0.0:
                raise ValueError("lr_sched.lr() must return a finite, non-negative value")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
            lr_sched=lr_sched,
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                lr_sched = group["lr_sched"]
                if lr_sched is None:
                    alpha = group["lr"]
                else:
                    alpha = float(lr_sched.lr())
                if not math.isfinite(alpha) or alpha < 0.0:
                    raise ValueError(
                        "lr_sched.lr() must return a finite, non-negative value"
                    )
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                correct_bias = group["correct_bias"]

                # Initialize state on first step
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)     # m_0
                    state["exp_avg_sq"] = torch.zeros_like(p.data)  # v_0

                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # 1. Update biased first and second moment estimates (Algorithm 1, lines 5-6)
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 2. Compute bias-corrected estimates (Algorithm 1, lines 7-8)
                if correct_bias:
                    m_hat = m / (1.0 - beta1 ** t)
                    v_hat = v / (1.0 - beta2 ** t)
                else:
                    m_hat = m
                    v_hat = v

                # 3. Update parameters (Algorithm 1, line 9)
                p.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-alpha)

                # 4. Decoupled weight decay applied after the Adam update
                if weight_decay != 0.0:
                    p.data.add_(p.data, alpha=-alpha * weight_decay)

        return loss
