import jax
import jax.numpy as jnp
import flax.linen as nn

from einops.einops import rearrange
from typing import Optional, Callable, List, TypeAlias, Union, Tuple, Literal, Any, Dict, cast

ModuleType : TypeAlias = Union[nn.Module, Callable[..., jnp.ndarray]]
ModuleList : TypeAlias = List[ModuleType]
Args : TypeAlias = Dict[str, jnp.ndarray]

class Identity(nn.Module):
    def __call__(self, args : Args) -> Args:
        return args

class Linear(nn.Module):
    features : int
    bias : bool = False

    @nn.compact
    def __call__(self, args : Args, key : str = "array") -> Args:
        args[key] = nn.Dense(features=self.features, use_bias=self.bias)(args[key])
        return args
    
class MLP(nn.Module):
    features : int
    hidden_features : Optional[int] = None
    bias : bool = False
    activation : ModuleType = nn.relu

    def setup(self):
        self.hidden = self.hidden_features if self.hidden_features else self.features

    @nn.compact
    def __call__(self, args : Args, key : str = "array") -> Args:
        args = Linear(features=self.hidden, bias=self.bias)(args)
        args[key] = self.activation(args[key])
        args = Linear(features=self.features, bias=self.bias)(args)
        return args
    
class Conv(nn.Module):
    features : int
    kernel_size : Union[int, Tuple[int, int]] = (3, 3)
    strides : Union[int, Tuple[int, int]] = (1, 1)
    padding : Literal["SAME", "VALID"] = "SAME"

    @nn.compact
    def __call__(self, args : Args, key : str = "array") -> Args:
        args[key] = nn.Conv(self.features, self.kernel_size, self.strides, self.padding)(args[key])
        return args
    
class Residual(nn.Module):
    module : ModuleType

    @nn.compact
    def __call__(self, args : Args, key : str = "array") -> Args:
        args[key] = self.module(args) + args[key]
        return args
    
class ScaleConv(nn.Module):
    features: int
    scale: float
    method: Literal["nearest", "lanczos3", "lanczos5"]
    antialias: bool = True

    def setup(self):
        self._scale : Tuple[float, float] = (self.scale, self.scale)

    @nn.compact
    def __call__(self, args : Args, key : str = "array") -> Args:
        scale_height, scale_width = self._scale
        array = args[key]
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

        return Conv(self.features)(args)
    
class Norm(nn.Module):
    type : Literal["layer", "batch", "group"] = "layer"

    @nn.compact
    def __call__(self, args : Args, key : str = "array") -> jnp.ndarray:
        if self.norm == "batch":
            norm_fn = nn.BatchNorm()
        elif self.norm == "group":
            norm_fn = nn.GroupNorm()
        else:
            norm_fn = nn.LayerNorm()
        return norm_fn(args[key])
    
class ConvBlock(nn.Module):
    features : int
    norm_type : Literal["layer", "batch", "group"] = "layer"
    activation : ModuleType = nn.relu
    
    @nn.compact
    def __call__(self, args : Args, scale_key : str = "scale", shift_key : str = "shift") -> jnp.ndarray:
        array = Conv(self.features)(args)
        array = Norm(self.norm_type)(args)
        if args.get(scale_key) and args.get(shift_key):
            scale, shift = args[scale_key], args[shift_key]
            array = array * (scale + 1) + shift

        array = self.activation(array)
        return array
    
class TimeShiftBlock(nn.Module):
    features : int
    embed_dim : Optional[int] = None
    hidden_features : Optional[int] = None
    bias : bool = False
    activation : ModuleType = nn.relu
    norm_type : Literal["layer", "batch", "group"] = "layer"

    @nn.compact
    def __call__(self, args : Args, embed_key : str = "embedding", scale_key : str = "scale", shift_key : str = "shift"):
        if args.get("embedding"):
            time_emb =  MLP(self.features * 2, self.hidden_features, self.bias, self.activation)({"array" : args["embedding"]})
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale, shift = jnp.split(args[embed_key], 2, axis=1)
            args[scale_key], args[shift_key] = scale, shift
            array = ConvBlock(self.features, self.norm_type, self.activation)(args, scale_key=scale_key, shift_key=shift_key)
        array = ConvBlock(self.features, self.norm_type, self.activation)(args)
        return array