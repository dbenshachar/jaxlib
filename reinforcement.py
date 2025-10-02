from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from flax import struct
from flax import core
import flax
from flax.training import train_state
import flax.linen as nn
import optax
from typing import Literal, Callable, Any
from training import *
import numpy as np
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

def make_env(seed, name : str, max_episode_steps : int = 500):
    def thunk():
        env = gym.make(name, max_episode_steps=max_episode_steps)
        env.reset(seed=seed)
        return env
    return thunk

def parallelize_env(num_envs, name : str, max_episode_steps : int = 500):
    env = SyncVectorEnv([make_env(seed, name, max_episode_steps=max_episode_steps) for seed in range(num_envs)])
    return env

@struct.dataclass
class Batch:
    obs : jax.Array
    actions : jax.Array
    rewards : jax.Array
    next_obs : jax.Array
    dones : jax.Array

class ReplayBuffer(ABC):
    @abstractmethod
    def __init__(self, capacity: int, obs_dim: int):
        ...
    
    @abstractmethod
    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        pass

    @abstractmethod
    def sample(self, batch_size: int, rng: jax.Array) -> Batch:
        ...

    def __len__(self) -> int:
        ...


class CircularReplayBuffer(ReplayBuffer):    
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.ptr = 0
        self.size = 0
        
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
    
    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, rng: jax.Array) -> Batch:
        if self.size < batch_size:
            raise ValueError(f"Can't sample {batch_size} from buffer with only {self.size} samples")
        
        indices = jax.random.choice(rng, self.size, shape=(batch_size,), replace=False)
        indices = np.array(indices)
        
        return Batch(
            obs=jnp.array(self.obs[indices]),
            actions=jnp.array(self.actions[indices]),
            rewards=jnp.array(self.rewards[indices]),
            next_obs=jnp.array(self.next_obs[indices]),
            dones=jnp.array(self.dones[indices])
        )
    
    def __len__(self):
        return self.size

@dataclass
class DoubleDeepQNetwork:
    model_constructor : Callable[[int], nn.Module]
    obs_dim : int
    n_actions : int
    lr : float = 3e-4
    optimizer_type : Literal["adam", "adamw", "sgd"] = "adam"
    decay_type : Literal["constant", "decay"] = "constant"
        
    @struct.dataclass
    class DDQNState(train_state.TrainState):
        target_params: core.FrozenDict

    def create_state(self, rng : jax.Array) -> DDQNState:
        self.model = self.model_constructor(self.n_actions)
        params = self.model.init(rng, jnp.zeros((1, self.obs_dim)))["params"]
        tx = make_optimizer(self.optimizer_type, self.decay_type, self.lr)
        return self.DDQNState.create(apply_fn=self.model.apply, params=params, tx=tx, target_params=params)
    
    @staticmethod
    def soft_update(target, online, tau : float):
        return jax.tree_util.tree_map(lambda t, o: (1.0 - tau) * t + tau * o, target, online)
    
    @staticmethod
    @jax.jit
    def train_step(state: DDQNState, batch : Batch, gamma: float, tau: float) -> DDQNState:
        def loss_fn(params):
            q = state.apply_fn({"params": params}, batch.obs)
            q_a = jnp.take_along_axis(q, batch.actions[..., None], axis=1).squeeze(-1)
            next_q_online = state.apply_fn({"params": params}, batch.next_obs)
            next_act = jnp.argmax(next_q_online, axis=1)
            next_q_target = state.apply_fn({"params": state.target_params}, batch.next_obs)
            next_q = jnp.take_along_axis(next_q_target, next_act[..., None], axis=1).squeeze(-1)
            target = batch.rewards + gamma * (1.0 - batch.dones) * next_q
            loss = jnp.mean(optax.huber_loss(q_a, target, delta=1.0))
            return loss

        grads = jax.grad(loss_fn)(state.params)
        updates, new_opt_state = state.tx.update(grads, state.opt_state, params=state.params)
        new_params = optax.apply_updates(state.params, updates)
        new_target = DoubleDeepQNetwork.soft_update(state.target_params, new_params, tau)
        new_state = state.replace(params=new_params, opt_state=new_opt_state, target_params=new_target)
        return new_state

    def select_action_with_n(self, state: DDQNState, obs_np, eps: float, rng : jax.Array) -> jax.Array:
        obs = jnp.asarray(obs_np)
        n = obs.shape[0]
        u_key, r_key = jax.random.split(rng, 2)
        u = jax.random.uniform(key=u_key, shape=(n,))
        q = state.apply_fn({"params": state.params}, obs)
        greedy = jnp.argmax(q, axis=1)
        random_a = jax.random.randint(key=r_key, shape=(n,), minval=0, maxval=self.n_actions)
        mask = u < eps
        return jnp.where(mask, random_a, greedy)
    
    def train(
        self,
        env,
        buffer : ReplayBuffer,
        total_timesteps: int = 100_000,
        batch_size: int = 32,
        learning_starts: int = 1000,
        train_freq: int = 4,
        target_update_freq: int = 1000,
        gamma: float = 0.99,
        tau: float = 1.0,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 80_000,
        rng_seed: int = 42,
        eval_freq: int = 10_000,
        eval_episodes: int = 5
        ) -> DDQNState:
        rng = jax.random.PRNGKey(rng_seed)
        rng, init_rng = jax.random.split(rng)
        state = self.create_state(init_rng)

        obs, _ = env.reset()
        obs = jnp.asarray(obs)
        num_envs = obs.shape[0]
        episode_reward = jnp.zeros((num_envs,), dtype=jnp.float32)
        episode_count = 0
        episode_rewards = []

        for step in range(total_timesteps):
            eps = max(eps_end, eps_start - (eps_start - eps_end) * step / eps_decay_steps)
            rng, action_rng = jax.random.split(rng)
            action = self.select_action_with_n(state, obs, eps, action_rng)

            next_obs, reward, terminated, truncated, _ = env.step(action.tolist())
            next_obs = jnp.asarray(next_obs)
            reward = jnp.asarray(reward, dtype=jnp.float32)
            terminated = jnp.asarray(terminated)
            truncated = jnp.asarray(truncated)
            done = jnp.logical_or(terminated, truncated)

            for i in range(len(done)):
                buffer.add(
                    obs[i], # pyright: ignore[reportArgumentType]
                    int(action[i].item()),
                    float(reward[i].item()),
                    next_obs[i], # pyright: ignore[reportArgumentType]
                    bool(done[i].item())
                )

            episode_reward = episode_reward + reward
            for i in range(len(done)):
                if bool(done[i].item()):
                    episode_rewards.append(float(episode_reward[i].item()))
                    episode_count += 1

            episode_reward = jnp.where(done, 0.0, episode_reward)
            obs = next_obs

            if step % 1000 == 0:
                if episode_rewards:
                    recent_rewards = episode_rewards[-100:]
                    avg_reward = float(jnp.mean(jnp.array(recent_rewards)).item())
                else:
                    avg_reward = 0.0
                print(f"Step {step} | Episodes: {episode_count} | Avg Reward: {avg_reward:.2f} | Eps: {eps:.3f}")
                
            if step % eval_freq == 0 and step > 0:
                rng, eval_rng = jax.random.split(rng)
                eval_reward = self.evaluate(env, state, eval_rng, eval_episodes)
                print(f"Eval Step {step}: {eval_reward:.2f}")
                
            if step >= learning_starts and step % train_freq == 0 and len(buffer) >= batch_size:
                rng, sample_rng = jax.random.split(rng)
                batch = buffer.sample(batch_size, sample_rng)
                update_tau = tau if tau < 1.0 else (1.0 if step % target_update_freq == 0 else 0.0)
                state = DoubleDeepQNetwork.train_step(state, batch, gamma, update_tau)
                
        return state

    def evaluate(self, env, state, rng, num_episodes: int = 5) -> float:
        obs, _ = env.reset()
        obs = jnp.asarray(obs)
        num_envs = obs.shape[0]
        ep_returns = jnp.zeros((num_envs,), dtype=jnp.float32)
        completed = []
        while len(completed) < num_episodes:
            action = self.select_action_with_n(state, obs, 0.0, rng)
            obs, reward, terminated, truncated, _ = env.step(action.tolist())
            obs = jnp.asarray(obs)
            reward = jnp.asarray(reward, dtype=jnp.float32)
            done = jnp.logical_or(jnp.asarray(terminated), jnp.asarray(truncated))
            ep_returns = ep_returns + reward
            for i in range(num_envs):
                if bool(done[i].item()):
                    completed.append(float(ep_returns[i].item()))
                    ep_returns = ep_returns.at[i].set(0.0)
        return float(jnp.mean(jnp.array(completed[:num_episodes])).item())