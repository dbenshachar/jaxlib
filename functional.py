from typing import Dict
import jax
import jax.numpy as jnp

from modules import *
from flax.training import train_state
import optax

def make_optimizer(optimizer_type : Literal["adam", "adamw", "sgd"], decay_type : Literal["constant", "decay"],
                lr : float = 3e-4, transition_steps : int = 2000, decay_rate : float = 0.95,
                b1 : float = 0.9, b2 : float = 0.999, eps : float = 1e-8) -> optax.GradientTransformationExtraArgs:
    if decay_type == "decay" and lr and transition_steps and decay_rate:
        schedule = optax.exponential_decay(
            lr, transition_steps, decay_rate
        )
    else:
        schedule = optax.constant_schedule(lr)
    if optimizer_type == "adam":
        optimizer = optax.adam(schedule, b1, b2, eps)
    elif optimizer_type == "adamw":
        optimizer = optax.adamw(schedule, b1, b2, eps)
    elif optimizer_type == "sgd":
        optimizer = optax.sgd(schedule)
    return optimizer

def create_train_state(model : nn.Module, optimizer : optax.GradientTransformationExtraArgs, shape : List[int], rng : int = 42) -> train_state.TrainState:
    params = model.init(jax.random.PRNGKey(rng), jnp.ones(shape))["params"]
    return train_state.TrainState(0, apply_fn=model.apply, params=params, tx=optimizer, opt_state=optimizer.init(params))