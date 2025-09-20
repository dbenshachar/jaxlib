import jax.numpy as jnp
import flax.linen as nn

from typing import Optional, Callable, List, TypeAlias, Union

ModuleList : TypeAlias = List[Union[nn.Module, Callable[..., jnp.ndarray]]]

class Identity():
    def __init__(self, *args): # type: ignore
        ...

    def __call__(self, array : jnp.ndarray, *args) -> jnp.ndarray: # type: ignore
        return array


class Linear(nn.Module):
    features : int
    bias : bool = False

    def __setup__(self):
        self.module = nn.Dense(features=self.features, use_bias=self.bias)

    def __call__(self, array : jnp.ndarray) -> jnp.ndarray:
        return self.module(array)
    
class MLP(nn.Module):
    features : int
    hidden_features : Optional[int] = None
    bias : bool = False
    activation : Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None

    def __setup__(self):
        activation = self.activation if self.activation else Identity()
        self.hidden_features = self.hidden_features if self.hidden_features else self.features
        self.layers : ModuleList = [
            Linear(features=self.hidden_features, bias=self.bias),
            activation,
            Linear(features=self.features),
        ]

    def __call__(self, array : jnp.ndarray) -> jnp.ndarray:
        for module in self.layers:
            array = module(array)
        return array