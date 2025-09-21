import jax
import jax.numpy as jnp
import flax.linen as nn

from einops.einops import rearrange
from typing import Optional, Callable, List, Sequence, TypeAlias, Union, Tuple, Literal, Any, Dict, cast

ModuleType : TypeAlias = Union[nn.Module, Callable[..., jnp.ndarray]]
ModuleList : TypeAlias = List[ModuleType]
Args : TypeAlias = Dict[str, jnp.ndarray]

class Identity(nn.Module):
    """
    Identity module.
    
    Inputs:
    - args: dict, must contain key "array" with jnp.ndarray of shape (B, ...).
    
    Output:
    - Returns the same args dictionary with no modifications.
    
    Example:
    >>> import jax.numpy as jnp
    >>> args = {"array": jnp.array([[1,2],[3,4]])}  # shape (2,2)
    >>> out = Identity()(args)
    # Updated args: {"array": jnp.ndarray of shape (2,2)}
    """

    def __call__(self, args : Args) -> Args:
        return args

class Linear(nn.Module):
    """Linear layer applying a Dense transformation.
    
    Inputs:
    - args: dict with key "array" containing a jnp.ndarray of shape (B, *, F_in) where F_in is the input feature dimension.
    
    Updated args:
    - "array": jnp.ndarray of shape (B, *, features), where features is set by the module attribute.
    
    Example:
    >>> import jax.numpy as jnp
    >>> args = {"array": jnp.ones((2, 10))}  # shape (2,10)
    >>> out = Linear(features=5, bias=True)(args)
    # Updated args: {"array": jnp.ndarray of shape (2, 5)}
    """
    features : int
    bias : bool = False

    @nn.compact
    def __call__(self, args : Args, key : str = "array") -> Args:
        args[key] = nn.Dense(features=self.features, use_bias=self.bias)(args[key])
        return args
    
class MLP(nn.Module):
    """
    MLP module: A multi-layer perceptron using two Linear layers with an activation function between them.
    
    Inputs:
      - args: dict with key "array" containing a jnp.ndarray of shape (B, F_in), where B is the batch size and F_in is the input feature dimension.
    
    Updated args:
      - "array": After the first Linear layer, the shape becomes (B, hidden) where hidden = hidden_features (or features if not provided).
                   After activation and the second Linear layer, the shape becomes (B, features).
    
    Example:
      >>> import jax.numpy as jnp
      >>> args = {"array": jnp.ones((2, 10))}  # input shape: (2,10)
      >>> out = MLP(features=5, hidden_features=7)(args)
      # Updated args: {"array": jnp.ndarray of shape (2,5)}
    """
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
    """Conv layer applying a convolution operation.
    
    Inputs:
      - args: dict with key "array" containing a jnp.ndarray of shape (B, H, W, C_in), where B is batch size, H is height, W is width, and C_in is the number of input channels.
    
    Updated args:
      - "array": jnp.ndarray with shape (B, H_out, W_out, features), where H_out and W_out depend on the kernel size, strides and padding.
    
    Example:
      >>> import jax.numpy as jnp
      >>> args = {"array": jnp.ones((2, 28, 28, 3))}  # input shape: (2,28,28,3)
      >>> out = Conv(features=16, kernel_size=(3,3), strides=(1,1))(args)
      # Updated args: {"array": jnp.ndarray of shape (2, H_out, W_out, 16)}
    """
    features : int
    kernel_size : Union[int, Tuple[int, int]] = (3, 3)
    strides : Union[int, Tuple[int, int]] = (1, 1)
    padding : Literal["SAME", "VALID"] = "SAME"

    @nn.compact
    def __call__(self, args : Args, key : str = "array") -> Args:
        args[key] = nn.Conv(self.features, self.kernel_size, self.strides, self.padding)(args[key])
        return args
    
class Residual(nn.Module):
    """Residual module that adds the output of a sub-module to the input, forming a skip connection.
    
    Inputs:
      - args: dict with key "array" containing a jnp.ndarray of shape (B, ...), where B is the batch size.
    
    Updated args:
      - "array": jnp.ndarray of the same shape as input, computed as module(args) + input array.
    
    Example:
      >>> import jax.numpy as jnp
      >>> def some_module(args):
      ...     args["array"] = args["array"] * 2
      ...     return args
      >>> args = {"array": jnp.ones((2, 10))}  # input shape: (2,10)
      >>> out = Residual(module=some_module)(args)
      # Updated args: {"array": jnp.ndarray of shape (2, 10)}
    """
    module : ModuleType

    @nn.compact
    def __call__(self, args : Args, key : str = "array") -> Args:
        args[key] = self.module(args) + args[key]
        return args
    
class ScaleConv(nn.Module):
    """ScaleConv module that performs image resizing before applying a convolution.
    
    Inputs:
      - args: dict with key "array" containing a jnp.ndarray of shape (B, H, W, C_in), where
              B is the batch size, H height, W width, and C_in number of input channels.
    
    Processing steps:
      1. Resizes the input array by a given scale factor along its height and width.
      2. Applies a convolution (using Conv) that updates the shape to (B, H_out, W_out, features).
    
    Updated args:
      - "array": jnp.ndarray updated after convolution, with shape (B, H_out, W_out, features),
                 where H_out and W_out are computed as H * scale and W * scale (if scaling occurs).
    
    Example:
      >>> import jax.numpy as jnp
      >>> args = {"array": jnp.ones((2, 32, 32, 3))}  # input shape: (2,32,32,3)
      >>> out = ScaleConv(features=16, scale=0.5, method="lanczos3", antialias=True)(args)
      # Updated args: {"array": jnp.ndarray of shape (2, H_out, W_out, 16)}
    """
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
            args[key] = array
        
        return Conv(self.features)(args)
    
class Norm(nn.Module):
    """Normalization module that normalizes inputs based on the specified type.
    
    Inputs:
      - args: dict with key "array" containing a jnp.ndarray of shape (B, ...), where B is the batch size.
    
    Updated args:
      - Returns a normalized jnp.ndarray with the same shape as the input.
    
    Example:
      >>> import jax.numpy as jnp
      >>> args = {"array": jnp.ones((2, 10))}  # input shape: (2,10)
      >>> normalized = Norm(type="layer")(args)
      # Updated output: jnp.ndarray of shape (2,10)
    """
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
    """ConvBlock module that applies convolution, normalization, optional scaling and shifting, and an activation function.
    
    Inputs:
      - args: dict with key "array" containing a jnp.ndarray of shape (B, H, W, C_in), where B is the batch size, H is height, W is width, and C_in is the number of input channels.
      - Additionally, may contain keys "scale" and "shift" with jnp.ndarray values to adjust the output if provided.
    
    Processing steps:
      1. Applies a convolution via the Conv module resulting in an output with shape modified according to convolution parameters.
      2. Applies a normalization (LayerNorm, BatchNorm, or GroupNorm) via the Norm module.
      3. If both "scale" and "shift" are provided in args, computes: output = normalized * (scale + 1) + shift.
      4. Applies the activation function.
    
    Returns:
      - A jnp.ndarray resulting from the activation, with shape determined by the convolution and normalization operations.
    
    Example:
      >>> import jax.numpy as jnp
      >>> args = {"array": jnp.ones((2, 28, 28, 3)), "scale": jnp.ones((2,1,1,3)), "shift": jnp.zeros((2,1,1,3))}  # input shape: (2,28,28,3)
      >>> out = ConvBlock(features=16, norm_type="layer", activation=jax.nn.relu)(args)
      # Updated args: {"array": jnp.ndarray of shape (2, H_out, W_out, 16)}
    """
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
    """TimeShiftBlock module that adjusts input based on time embeddings and applies ConvBlock transformations.
    
    Inputs:
      - args: dict with key "array" containing a jnp.ndarray of shape (B, H, W, C) where B is batch size, H and W are spatial dimensions, and C is channel depth.
      - Optionally, may contain key "embedding" with a jnp.ndarray of shape (B, E) representing time embeddings.
    
    Processing steps:
      - If "embedding" is provided in args, processes it through an MLP to generate time embeddings and rearranges them to shape (B, C', 1, 1).
      - Splits the embedding (assumed shape (B, 2*C')) into two parts along the channel dimension, which are assigned to keys specified by scale_key and shift_key.
      - Applies a ConvBlock using the updated args.
      - Applies an additional ConvBlock to produce the final output.
    
    Updated args:
      - "scale": jnp.ndarray of shape derived from splitting the embedding, typically (B, C')
      - "shift": jnp.ndarray of same shape as scale.
      - "array": The output array from ConvBlock will have shape (B, H_out, W_out, features) determined by convolution parameters.
    
    Example:
      >>> import jax.numpy as jnp
      >>> args = {"array": jnp.ones((2, 32, 32, 3)), "embedding": jnp.ones((2, 6))}  # input shape: (2,32,32,3)
      >>> out = TimeShiftBlock(features=3)(args)
      # Updated args: {"array": jnp.ndarray of shape (2, H_out, W_out, 3)}
    """
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
    
class PatchEmbedding(nn.Module):
    """PatchEmbedding module that converts an image into a sequence of patches.
    
    Inputs:
      - args: dict with key "array" containing a jnp.ndarray of shape (B, H, W, C), where B is the batch size,
              H is height, W is width, and C is the number of channels.
    
    Processing steps:
      1. Applies a convolution with kernel size and strides equal to the patch size, producing an output of shape
         (B, H_out, W_out, features) where H_out = H / patch_size[0] and W_out = W / patch_size[1].
      2. Reshapes the output to (B, H_out * W_out, features).
      3. If add_cls_token is True, prepends a learnable class token, changing the output shape to
         (B, 1 + H_out * W_out, features).
    
    Updated args:
      - "array": jnp.ndarray with shape (B, 1 + (H/patch_size[0])*(W/patch_size[1]), features) if add_cls_token
                 is True, otherwise (B, (H/patch_size[0])*(W/patch_size[1]), features).
    
    Example:
      >>> import jax.numpy as jnp
      >>> args = {"array": jnp.ones((2, 32, 32, 3))}  # input shape: (2,32,32,3)
      >>> out = PatchEmbedding(features=64, patch_size=(4, 4))(args)
      # Updated args: {"array": jnp.ndarray of shape (2, 1 + (32/4)*(32/4), 64) if add_cls_token is True}
    """
    features: int
    patch_size: Sequence[int]
    bias: bool = True
    add_cls_token: bool = True

    def setup(self):
        if isinstance(self.patch_size, int):
            self.patch_size = (self.patch_size, self.patch_size)

    @nn.compact
    def __call__(self, args: dict, key: str = "array") -> dict:
        key = key or self.key
        ph, pw = self.patch_size
        args[key] = nn.Conv(
            features=self.features,
            kernel_size=(ph, pw),
            strides=(ph, pw),
            use_bias=self.bias,
            name="proj",
        )(args[key])

        B, H, W, D = args[key].shape
        args[key] = jnp.reshape(args[key], (B, H * W, D))

        if self.add_cls_token:
            cls = self.param("cls", nn.initializers.zeros, (1, 1, D))
            args[key] = jnp.concatenate([jnp.tile(cls, (B, 1, 1)), args[key]], axis=1)

        return args