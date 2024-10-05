import math

import torch


class CosineAnnealing:
    def __init__(self, warmup_steps: int, max_steps: int):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def __call__(self, current_step: int) -> float:
        if current_step < self.warmup_steps:
            return current_step / self.warmup_steps
        return 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (current_step - self.warmup_steps)
                / (self.max_steps - self.warmup_steps)
            )
        )


def get_cosine_with_warmup(
    optimizer: torch.optim.Optimizer, warmup_steps: int, max_steps: int
):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, CosineAnnealing(warmup_steps=warmup_steps, max_steps=max_steps)
    )


def get_scheduler(name: str, **params):
    name2callable = {"CosineAnnealing": get_cosine_with_warmup}

    if name in name2callable:
        return name2callable[name](**params)

    raise KeyError(f"Implement {name} scheduler")


def get_optimizer(name: str, **params) -> torch.optim.Optimizer:
    name2callable = {"Adam": torch.optim.Adam}

    if name in name2callable:
        return name2callable[name](**params)

    raise KeyError(f"Implement {name} optimizer")
