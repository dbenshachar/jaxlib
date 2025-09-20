import jax
import jax.numpy as jnp
import flax.linen as nn

from einops.einops import rearrange
from typing import Optional, Callable, List, TypeAlias, Union, Tuple, Literal, Any, cast

ModuleType : TypeAlias = Union[nn.Module, Callable[..., jnp.ndarray]]
ModuleList : TypeAlias = List[ModuleType]

class Identity(nn.Module):
    def __init__(self, *args : Any):
        ...

    def __call__(self, array : jnp.ndarray, *args : Any) -> jnp.ndarray:
        return array


class Linear(nn.Module):
    features : int
    bias : bool = False

    def setup(self):
        self.module = nn.Dense(features=self.features, use_bias=self.bias)

    def __call__(self, array : jnp.ndarray) -> jnp.ndarray:
        return self.module(array)
    
class MLP(nn.Module):
    features : int
    hidden_features : Optional[int] = None
    bias : bool = False
    activation : ModuleType = nn.relu

    def setup(self):
        self.hidden_features = self.hidden_features if self.hidden_features else self.features
        self.layers : ModuleList = [
            Linear(features=self.hidden_features, bias=self.bias),
            self.activation,
            Linear(features=self.features),
        ]

    def __call__(self, array : jnp.ndarray) -> jnp.ndarray:
        for module in self.layers:
            array = module(array)
        return array
    
class Conv(nn.Module):
    features : int
    kernel_size : Union[int, Tuple[int, int]] = (3, 3)
    strides : Union[int, Tuple[int, int]] = (1, 1)
    padding : Literal["SAME", "VALID"] = "SAME"

    def setup(self):
        self.module = nn.Conv(self.features, self.kernel_size, self.strides, self.padding)
    
    def __call__(self, array : jnp.ndarray) -> jnp.ndarray:
        return self.module(array)
    
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
        self.conv = Conv(self.features)
        self._scale : Tuple[float, float] = (self.scale, self.scale)

    def __call__(self, array: jnp.ndarray) -> jnp.ndarray:
        scale_height, scale_width = self._scale
        batch, height, width, channel = array.shape

        if scale_height != 1.0 or scale_width != 1.0:
            height = height * scale_height
            width = width * scale_width
            assert height.is_integer() and width.is_integer()
            height, width = int(height), int(width)

            array = cast(jnp.ndarray, 
                        jax.image.resize( # pyright: ignore[reportUnknownMemberType]
                            array,
                            shape=(batch, height, width, channel),
                            method=self.method,
                            antialias=self.antialias)
                        )

        return self.conv(array)
    
class ConvBlock(nn.Module):
    features : int
    norm : ModuleType = nn.LayerNorm()
    activation : ModuleType = nn.relu

    def setup(self):
        self.conv = Conv(self.features)
    
    def __call__(self, array : jnp.ndarray, scale_shift : Optional[List[jnp.ndarray]] = None) -> jnp.ndarray:
        array = self.conv(array)
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

    def setup(self):
        self.mlp : Optional[ModuleType] = None
        if self.embed_dim:
            self.mlp = MLP(self.features, self.hidden_features, self.bias, self.activation)
        self.scale_shift = ConvBlock(self.features, self.norm, self.activation)
        self.conv = ConvBlock(self.features, self.norm, self.activation)

    def __call__(self, array : jnp.ndarray, embedding : Optional[jnp.ndarray] = None):
        if self.mlp and embedding:
            time_emb = self.mlp(embedding)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = jnp.split(time_emb, 2, axis=1)
            array = self.scale_shift(array, scale_shift=scale_shift)
        array = self.conv(array)
        return array