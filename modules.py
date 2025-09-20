import jax
import jax.numpy as jnp
import flax.linen as nn

from einops.einops import rearrange
from typing import Optional, Callable, List, TypeAlias, Union, Tuple, Literal, Any, cast

ModuleType : TypeAlias = Union[nn.Module, Callable[..., jnp.ndarray]]
ModuleList : TypeAlias = List[ModuleType]

class Identity(nn.Module):
    def __call__(self, array : jnp.ndarray, *args : Any) -> jnp.ndarray:
        return array


class Linear(nn.Module):
    features : int
    bias : bool = False

    @nn.compact
    def __call__(self, array : jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(features=self.features, use_bias=self.bias)(array)
    
class MLP(nn.Module):
    features : int
    hidden_features : Optional[int] = None
    bias : bool = False
    activation : ModuleType = nn.relu

    def setup(self):
        self.hidden = self.hidden_features if self.hidden_features else self.features

    @nn.compact
    def __call__(self, array : jnp.ndarray) -> jnp.ndarray:
        array = Linear(features=self.hidden, bias=self.bias)(array)
        array = self.activation(array)
        array = Linear(features=self.features, bias=self.bias)(array)
        return array
    
class Conv(nn.Module):
    features : int
    kernel_size : Union[int, Tuple[int, int]] = (3, 3)
    strides : Union[int, Tuple[int, int]] = (1, 1)
    padding : Literal["SAME", "VALID"] = "SAME"

    @nn.compact
    def __call__(self, array : jnp.ndarray) -> jnp.ndarray:
        return nn.Conv(self.features, self.kernel_size, self.strides, self.padding)(array)
    
class Residual(nn.Module):
    module : ModuleType

    @nn.compact
    def __call__(self, array : jnp.ndarray, *args : Any, **kwargs : Any) -> jnp.ndarray:
        return self.module(array, *args, **kwargs) + array
    
class ScaleConv(nn.Module):
    features: int
    scale: float
    method: Literal["nearest", "linear", "cubic", "lanczos3", "lanczos5"]
    antialias: bool = True

    def setup(self):
        self._scale : Tuple[float, float] = (self.scale, self.scale)

    @nn.compact
    def __call__(self, array: jnp.ndarray) -> jnp.ndarray:
        scale_height, scale_width = self._scale
        batch, height, width, channel = array.shape

        if scale_height != 1.0 or scale_width != 1.0:
            height = height * scale_height
            width = width * scale_width
            assert height.is_integer() and width.is_integer()
            height, width = int(height), int(width)

            array = cast(jnp.ndarray, 
                        jax.image.resize(
                            array,
                            shape=(batch, height, width, channel),
                            method=self.method,
                            antialias=self.antialias)
                        )

        return Conv(self.features)(array)
    
class ConvBlock(nn.Module):
    features : int
    norm : ModuleType = nn.LayerNorm()
    activation : ModuleType = nn.relu
    
    @nn.compact
    def __call__(self, array : jnp.ndarray, scale_shift : Optional[List[jnp.ndarray]] = None) -> jnp.ndarray:
        array = Conv(self.features)(array)
        array = self.norm(array)
        if scale_shift:
            scale, shift = scale_shift
            array = array * (scale + 1) + shift

        array = self.activation(array)
        return array
    
class TimeShiftBlock(nn.Module):
    features : int
    embed_dim : Optional[int] = None
    hidden_features : Optional[int] = None
    bias : bool = False
    activation : ModuleType = nn.relu
    norm : ModuleType = nn.LayerNorm()

    @nn.compact
    def __call__(self, array : jnp.ndarray, embedding : Optional[jnp.ndarray] = None):
        if self.mlp and embedding:
            time_emb =  MLP(self.features, self.hidden_features, self.bias, self.activation)(embedding)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = jnp.split(time_emb, 2, axis=1)
            array = ConvBlock(self.features, self.norm, self.activation)(array, scale_shift)
        array = ConvBlock(self.features, self.norm, self.activation)(array)
        return array