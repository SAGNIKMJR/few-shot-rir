import torch.nn as nn


class UniformContextSampler(nn.Module):
    def __init__(
        self,
        actor_critic,
    ):
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, *x):
        raise NotImplementedError

    def update(self, rollouts):
        raise NotImplementedError

    def before_backward(self, loss):
        pass

    def after_backward(self, loss):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass
