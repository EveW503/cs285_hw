"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.mlp = nn.Sequential(
            nn.Linear(state_dim,hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0],hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1],hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2],chunk_size*action_dim)
        )

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        fn = nn.MSELoss()
        batch_size = state.shape[0]
        pred_n = self.mlp(state)
        pred = torch.reshape(
            pred_n,
            (batch_size,self.chunk_size,
            self.action_dim)
        )
        loss = fn(pred,action_chunk)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        pred_n = self.mlp(state)
        pred = torch.reshape(
            pred_n,
            (batch_size,self.chunk_size,
            self.action_dim)
            )
        return pred


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

        self.mlp = nn.Sequential(
            nn.Linear(state_dim+1+chunk_size*action_dim,hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0],hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1],hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2],chunk_size*action_dim)
        )

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        
        batch_size = state.shape[0]
        action_noise = torch.randn_like(action_chunk)
        tau = torch.rand(
            (batch_size,1,1),
            device=state.device,
            dtype=state.dtype,
        )

        a_tau = (1 - tau)*action_noise + tau*action_chunk
        vectory = action_chunk - action_noise

        mlp_input = torch.cat(
            [
                state,
                tau.reshape(batch_size,1),
                a_tau.flatten(start_dim=1)
            ],
            dim=1,
        )

        fn = nn.MSELoss()

        pred_n = self.mlp(mlp_input)
        pred = pred_n.reshape_as(action_chunk)

        loss = fn(pred,vectory)

        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
       batch_size = state.shape[0]

    # 从标准正态噪声开始
       actions = torch.randn(
        (batch_size, self.chunk_size, self.action_dim),
        device=state.device,
        dtype=state.dtype,
    )

       dt = 1.0 / num_steps

       for i in range(num_steps):
        tau = torch.full(
            (batch_size, 1),
            i * dt,
            device=state.device,
            dtype=state.dtype,
        )

        net_input = torch.cat(
            [
                state,
                tau,
                actions.flatten(start_dim=1),
            ],
            dim=1,
        )

        velocity = self.mlp(net_input).reshape_as(actions)

        # Euler 积分：根据当前速度向前走一步
        actions = actions + dt * velocity

       return actions  # (B, chunk_size, action_dim)


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
