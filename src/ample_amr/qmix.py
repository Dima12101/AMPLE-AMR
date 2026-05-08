"""QMIX-based node operation mode controller."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import torch
from torch import nn

from .allocation import BaseAllocator
from .config import QMIXConfig, WarehouseScenarioConfig
from .environment import SimulationEnvironment
from .utils import ensure_directory, set_global_seed


class FixedModeController:
    """Deterministic baseline with the same operation mode on every node."""

    def __init__(self, mode_name: str) -> None:
        self.mode_name = mode_name

    def select_modes(
        self,
        observations: list[list[float]],
        global_state: list[float],
        node_ids: list[str],
        explore: bool = False,
    ) -> tuple[dict[str, str], float]:
        return ({node_id: self.mode_name for node_id in node_ids}, 0.0)


class RandomModeController:
    """Random baseline over available operation modes."""

    def __init__(self, mode_names: list[str], seed: int) -> None:
        self.mode_names = mode_names
        self.rng = np.random.default_rng(seed)

    def select_modes(
        self,
        observations: list[list[float]],
        global_state: list[float],
        node_ids: list[str],
        explore: bool = False,
    ) -> tuple[dict[str, str], float]:
        return ({node_id: str(self.rng.choice(self.mode_names)) for node_id in node_ids}, 0.0)


class AgentNetwork(nn.Module):
    """Shared per-node action-value network."""

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.model(observations)


class MixingNetwork(nn.Module):
    """QMIX mixing network with state-conditioned positive weights."""

    def __init__(self, num_agents: int, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_agents * hidden_dim))
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_w2 = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        batch_size = agent_qs.shape[0]
        w1 = torch.abs(self.hyper_w1(states)).view(batch_size, self.num_agents, self.hidden_dim)
        b1 = self.hyper_b1(states).view(batch_size, 1, self.hidden_dim)
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        hidden = torch.relu(hidden)
        w2 = torch.abs(self.hyper_w2(states)).view(batch_size, self.hidden_dim, 1)
        b2 = self.hyper_b2(states).view(batch_size, 1, 1)
        mixed = torch.bmm(hidden, w2) + b2
        return mixed.view(batch_size)


@dataclass
class ReplayTransition:
    """Transition stored in the QMIX replay buffer."""

    observations: np.ndarray
    state: np.ndarray
    actions: np.ndarray
    reward: float
    next_observations: np.ndarray
    next_state: np.ndarray
    done: float


class ReplayBuffer:
    """Simple uniform replay buffer."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: deque[ReplayTransition] = deque(maxlen=capacity)

    def append(self, transition: ReplayTransition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[ReplayTransition]:
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[index] for index in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class QMIXModeController:
    """QMIX policy that chooses edge node operation modes only."""

    def __init__(
        self,
        scenario: WarehouseScenarioConfig,
        qmix_config: QMIXConfig,
        checkpoint_path: str | Path,
        seed: int,
        load_checkpoint: bool = True,
    ) -> None:
        set_global_seed(seed)
        torch.set_num_threads(1)
        self.scenario = scenario
        self.qmix_config = qmix_config
        self.checkpoint_path = Path(checkpoint_path)
        self.seed = seed
        self.mode_names = list(self.scenario.operation_modes)
        self.num_agents = scenario.edge_nodes
        self.action_dim = len(self.mode_names)
        probe_env = SimulationEnvironment(scenario, seed)
        observations, global_state = probe_env.reset(seed)
        self.obs_dim = len(observations[0])
        self.state_dim = len(global_state)
        self.agent_network = AgentNetwork(self.obs_dim, qmix_config.hidden_dim, self.action_dim)
        self.target_agent_network = AgentNetwork(self.obs_dim, qmix_config.hidden_dim, self.action_dim)
        self.mixing_network = MixingNetwork(self.num_agents, self.state_dim, qmix_config.mixing_hidden_dim)
        self.target_mixing_network = MixingNetwork(self.num_agents, self.state_dim, qmix_config.mixing_hidden_dim)
        self.optimizer = torch.optim.Adam(
            list(self.agent_network.parameters()) + list(self.mixing_network.parameters()),
            lr=qmix_config.learning_rate,
        )
        self.replay_buffer = ReplayBuffer(qmix_config.replay_capacity)
        self.training_steps = 0
        self.epsilon = qmix_config.epsilon_start
        self.training_history: list[dict[str, float]] = []
        self.target_agent_network.load_state_dict(self.agent_network.state_dict())
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        if load_checkpoint and self.checkpoint_path.exists():
            self.load()

    def select_modes(
        self,
        observations: list[list[float]],
        global_state: list[float],
        node_ids: list[str],
        explore: bool = False,
    ) -> tuple[dict[str, str], float]:
        start = perf_counter()
        observation_tensor = torch.tensor(np.asarray(observations, dtype=np.float32))
        with torch.no_grad():
            q_values = self.agent_network(observation_tensor).cpu().numpy()
        actions: list[int] = []
        for index, node_id in enumerate(node_ids):
            if explore and float(np.random.random()) < self.epsilon:
                action = int(np.random.randint(self.action_dim))
            else:
                action = int(np.argmax(q_values[index]))
            actions.append(action)
        modes = {node_id: self.mode_names[action] for node_id, action in zip(node_ids, actions, strict=True)}
        return modes, (perf_counter() - start) * 1000.0

    def train(
        self,
        allocator: BaseAllocator,
        env_factory: Callable[[int], SimulationEnvironment],
        episodes: int | None = None,
    ) -> list[dict[str, float]]:
        """Train QMIX on the provided environment factory."""

        num_episodes = episodes or self.qmix_config.training_episodes
        for episode in range(num_episodes):
            env = env_factory(self.seed + episode)
            observations, global_state = env.reset(self.seed + episode)
            episode_reward = 0.0
            loss_values: list[float] = []
            done = False
            while not done:
                node_ids = [node.id for node in env.nodes]
                mode_selection, _ = self.select_modes(observations, global_state, node_ids, explore=True)
                action_indices = np.asarray([self.mode_names.index(mode_selection[node_id]) for node_id in node_ids], dtype=np.int64)
                transition = env.step(mode_selection, allocator, policy_time_ms=0.0, compute_pricing=False)
                replay_transition = ReplayTransition(
                    observations=np.asarray(observations, dtype=np.float32),
                    state=np.asarray(global_state, dtype=np.float32),
                    actions=action_indices,
                    reward=float(transition.reward),
                    next_observations=np.asarray(transition.next_observations, dtype=np.float32),
                    next_state=np.asarray(transition.next_global_state, dtype=np.float32),
                    done=float(transition.done),
                )
                self.replay_buffer.append(replay_transition)
                observations = transition.next_observations
                global_state = transition.next_global_state
                done = transition.done
                episode_reward += transition.reward
                self.training_steps += 1
                self._update_epsilon()
                loss = self._optimize()
                if loss is not None:
                    loss_values.append(loss)
                if self.training_steps % self.qmix_config.target_update_interval == 0:
                    self.target_agent_network.load_state_dict(self.agent_network.state_dict())
                    self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
            history_row = {
                "episode": float(episode),
                "reward": float(episode_reward),
                "loss": float(np.mean(loss_values)) if loss_values else 0.0,
                "epsilon": float(self.epsilon),
            }
            self.training_history.append(history_row)
        self.save()
        return self.training_history

    def save(self) -> None:
        """Persist the trained controller to disk."""

        ensure_directory(self.checkpoint_path.parent)
        payload = {
            "agent_network": self.agent_network.state_dict(),
            "mixing_network": self.mixing_network.state_dict(),
            "target_agent_network": self.target_agent_network.state_dict(),
            "target_mixing_network": self.target_mixing_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "training_history": self.training_history,
            "obs_dim": self.obs_dim,
            "state_dim": self.state_dim,
        }
        torch.save(payload, self.checkpoint_path)

    def load(self) -> None:
        """Load a saved checkpoint if available."""

        payload = torch.load(self.checkpoint_path, map_location="cpu")
        self.agent_network.load_state_dict(payload["agent_network"])
        self.mixing_network.load_state_dict(payload["mixing_network"])
        self.target_agent_network.load_state_dict(payload["target_agent_network"])
        self.target_mixing_network.load_state_dict(payload["target_mixing_network"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.epsilon = float(payload.get("epsilon", self.qmix_config.epsilon_end))
        self.training_steps = int(payload.get("training_steps", 0))
        self.training_history = list(payload.get("training_history", []))

    def _optimize(self) -> float | None:
        if len(self.replay_buffer) < self.qmix_config.batch_size:
            return None
        batch = self.replay_buffer.sample(self.qmix_config.batch_size)
        observations = torch.tensor(np.stack([item.observations for item in batch], axis=0), dtype=torch.float32)
        states = torch.tensor(np.stack([item.state for item in batch], axis=0), dtype=torch.float32)
        actions = torch.tensor(np.stack([item.actions for item in batch], axis=0), dtype=torch.int64)
        rewards = torch.tensor(np.asarray([item.reward for item in batch], dtype=np.float32))
        next_observations = torch.tensor(np.stack([item.next_observations for item in batch], axis=0), dtype=torch.float32)
        next_states = torch.tensor(np.stack([item.next_state for item in batch], axis=0), dtype=torch.float32)
        dones = torch.tensor(np.asarray([item.done for item in batch], dtype=np.float32))

        q_values = self.agent_network(observations.view(-1, self.obs_dim)).view(-1, self.num_agents, self.action_dim)
        chosen_q_values = torch.gather(q_values, 2, actions.unsqueeze(-1)).squeeze(-1)
        mixed_q = self.mixing_network(chosen_q_values, states)

        with torch.no_grad():
            next_q_values = self.target_agent_network(next_observations.view(-1, self.obs_dim)).view(-1, self.num_agents, self.action_dim)
            max_next_q_values = next_q_values.max(dim=2).values
            mixed_target_q = self.target_mixing_network(max_next_q_values, next_states)
            targets = rewards + self.qmix_config.gamma * (1.0 - dones) * mixed_target_q

        loss = nn.functional.mse_loss(mixed_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.agent_network.parameters()) + list(self.mixing_network.parameters()), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def _update_epsilon(self) -> None:
        progress = min(1.0, self.training_steps / max(1, self.qmix_config.epsilon_decay_steps))
        self.epsilon = self.qmix_config.epsilon_start + progress * (
            self.qmix_config.epsilon_end - self.qmix_config.epsilon_start
        )
