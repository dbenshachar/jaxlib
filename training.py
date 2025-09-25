from typing import Dict
import jax
import jax.numpy as jnp

from modules import *
from flax.training import train_state
import optax
import einops

from torch.utils.data import DataLoader

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

def create_train_state(model : nn.Module, optimizer : optax.GradientTransformationExtraArgs, sample_input : List[int], rng : int = 42) -> train_state.TrainState:
    params = model.init(jax.random.PRNGKey(rng), sample_input)["params"]
    return train_state.TrainState(0, apply_fn=model.apply, params=params, tx=optimizer, opt_state=optimizer.init(params))

def compute_metrics(logits: jnp.ndarray, labels: jnp.ndarray, loss: Optional[float] = None) -> Dict[str, float]:
    res = {}
    if loss is not None:
        res["loss"] = loss
    return res

def forward(params, apply_fn, batch, rng: Optional[jax.Array] = None, *args : Any, **kwargs : Any,) -> jnp.ndarray:
    if rng is None:
        logits = apply_fn({"params": params}, batch, *args, **kwargs)
    else:
        logits = apply_fn({"params": params}, batch, rngs={"dropout": rng}, *args, **kwargs)
    return logits

@jax.jit
def train_step(state: train_state.TrainState, data: jnp.ndarray, target : jnp.ndarray, rng: Optional[jax.Array] = None) -> Tuple[train_state.TrainState, Dict[str, float]]:
    def loss_fn(p) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        logits = forward(p, state.apply_fn, data, rng=rng)
        loss = loss_calc(logits, target)
        return loss, (logits, target)

    (loss, (logits, target)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, target, loss=loss)
    return state, metrics

def loss_calc(logits : jnp.ndarray, target : jnp.ndarray) -> jnp.ndarray:
    return (logits - target).mean()

@jax.jit
def eval_step(state: train_state.TrainState, batch: Dict[str, jnp.ndarray]):
    logits, target = forward(state.params, state.apply_fn, batch, rng=None)
    return compute_metrics(logits, target)

def train(state : train_state.TrainState, train_loader : DataLoader, epochs : int = 20, num_logs : int = 20, rng : jnp.ndarray = jax.random.PRNGKey(42)):
    losses = []
    log_every = len(train_loader) // num_logs

    for epoch in range(1, epochs+1):
        for step, batch in enumerate(train_loader):
            data, target = batch
            rng, step_rng = jax.random.split(rng)
            state, metrics = train_step(state, data, target, rng=step_rng)
            losses.append(metrics["loss"])

            if step % log_every == 0:
                print(f"Epoch: {epoch} | Step: {step+1} | Loss: {float(metrics['loss']):.2f}")
    return losses