import jax
import jax.numpy as jnp
import flax.linen as nn

from einops.einops import rearrange
from typing import Optional, Callable, List, Sequence, TypeAlias, Union, Tuple, Literal, Any, Dict, cast

ModuleType : TypeAlias = Union[nn.Module, Callable[..., jnp.ndarray]]
ModuleList : TypeAlias = List[ModuleType]

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

    def __call__(self, array: jnp.ndarray) -> jnp.ndarray:
        return array


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
    features: int
    bias: bool = False

    @nn.compact
    def __call__(self, array: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(features=self.features, use_bias=self.bias)(array)
    
class GeGLU(nn.Module):
    @nn.compact
    def __call__(self, array : jnp.ndarray) -> jnp.ndarray:
        features = array.shape[-1]
        array, gate = jnp.split(Linear(features * 2)(array), 2, axis=-1)
        array = nn.gelu(array)
        array = array * gate
        array = Linear(features)(array)
        return array

class LayerScale(nn.Module):
    init_value: float = 1e-5
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, array : jnp.ndarray) -> jnp.ndarray:
        features = array.shape[-1]
        gamma = self.param("gamma", nn.initializers.constant(self.init_value), (features,))
        gamma = gamma.astype(self.dtype)
        return array * gamma
    
class DropPath(nn.Module):
    rate: float = 0.0 

    @nn.compact
    def __call__(self, array : jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        if deterministic or self.rate == 0.0:
            return array
        keep = 1.0 - self.rate
        rng = self.make_rng("dropout")
        shape = (array.shape[0],) + (1,) * (array.ndim - 1)
        mask = jax.random.bernoulli(rng, p=keep, shape=shape)
        return array * mask / keep

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
    hidden_features: Optional[int] = None
    bias: bool = False
    activation: ModuleType = GeGLU()
    dropout_rate : float = 0.1

    @nn.compact
    def __call__(self, array: jnp.ndarray) -> jnp.ndarray:
        dim = array.shape[-1]
        hidden = self.hidden_features if self.hidden_features else dim * 4
        array = Linear(features=hidden, bias=self.bias)(array)
        array = self.activation(array)
        array = nn.Dropout(self.dropout_rate)(array)
        array = Linear(features=dim, bias=self.bias)(array)
        return array


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
    features: int
    kernel_size: Union[int, Tuple[int, int]] = (3, 3)
    strides: Union[int, Tuple[int, int]] = (1, 1)
    padding: Literal["SAME", "VALID"] = "SAME"

    @nn.compact
    def __call__(self, array: jnp.ndarray) -> jnp.ndarray:
        return nn.Conv(self.features, self.kernel_size, self.strides, self.padding)(array)

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
        self._scale: Tuple[float, float] = (self.scale, self.scale)

    @nn.compact
    def __call__(self, array: jnp.ndarray) -> jnp.ndarray:
        scale_height, scale_width = self._scale
        batch, height, width, channel = array.shape

        if scale_height != 1.0 or scale_width != 1.0:
            new_height = height * scale_height
            new_width = width * scale_width
            assert new_height.is_integer() and new_width.is_integer()
            new_height, new_width = int(new_height), int(new_width)
            array = cast(jnp.ndarray,
                         jax.image.resize(
                             array,
                             shape=(batch, new_height, new_width, channel),
                             method=self.method,
                             antialias=self.antialias)
                        )
        return Conv(self.features)(array)

class RMSNorm(nn.Module):
    eps: float = 1e-8
    param_dtype: Optional[jnp.dtype] = None

    @nn.compact
    def __call__(self, array : jnp.ndarray) -> jnp.ndarray:
        features = array.shape[-1]
        scale = self.param(
            "scale",
            nn.initializers.ones,
            (features,),
            array.dtype,
        )
        rms = jnp.sqrt(jnp.mean(jnp.square(array), axis=-1, keepdims=True) + self.eps)
        return (array / rms) * scale

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
    type: Literal["layer", "batch", "group", "rms"] = "rms"

    @nn.compact
    def __call__(self, array: jnp.ndarray) -> jnp.ndarray:
        if self.type == "batch":
            norm_fn = nn.BatchNorm()
        elif self.type == "group":
            norm_fn = nn.GroupNorm()
        elif self.type == "layer":
            norm_fn = nn.LayerNorm()
        else:
            norm_fn = RMSNorm()
        return norm_fn(array)


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
      >>> out = ConvBlock(features=16, norm_type="layer", activation=jax.nn.gelu)(args)
      # Updated args: {"array": jnp.ndarray of shape (2, H_out, W_out, 16)}
    """
    features: int
    norm_type: Literal["layer", "batch", "group"] = "layer"
    activation: ModuleType = GeGLU()

    @nn.compact
    def __call__(self, array: jnp.ndarray, scale: Optional[jnp.ndarray] = None, shift: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        array = Conv(self.features)(array)
        array = Norm(self.norm_type)(array)
        if scale is not None and shift is not None:
            array = array * (scale + 1) + shift
        return self.activation(array)


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
    patch_size: Union[Tuple[int, int], int]
    bias: bool = True
    add_cls_token: bool = True

    @nn.compact
    def __call__(self, array: jnp.ndarray) -> jnp.ndarray:
        if isinstance(self.patch_size, int):
            patch_size = (self.patch_size, self.patch_size)
        else:
            patch_size = self.patch_size

        ph, pw = patch_size
        projected = nn.Conv(
            features=self.features,
            kernel_size=(ph, pw),
            strides=(ph, pw),
            use_bias=self.bias,
            name="proj",
        )(array)

        B, H, W, D = projected.shape
        projected = jnp.reshape(projected, (B, H * W, D))

        if self.add_cls_token:
            cls = self.param("cls", nn.initializers.zeros, (1, 1, D))
            projected = jnp.concatenate([jnp.tile(cls, (B, 1, 1)), projected], axis=1)

        return projected
    
class Attention(nn.Module):
    num_heads: int
    hidden_dim: Optional[int] = None
    dropout_rate: float = 0.0
    deterministic: bool = True
    qkv_bias : bool = True
    rotary_base : int = 1_000
    cls_index : int = 0

    @nn.compact
    def __call__(self, array : jnp.ndarray, kv : Optional[jnp.ndarray] = None, apply_bias : Optional[jnp.ndarray] = None) -> jnp.ndarray:
        features = array.shape[-1]
        if not self.hidden_dim:
            hidden_dim = array.shape[-1] * self.num_heads
        else:
            hidden_dim = self.hidden_dim * self.num_heads

        if kv:
            q = Linear(hidden_dim)(array)
            k, v = jnp.split(Linear(hidden_dim * 2, bias=self.qkv_bias)(kv), 2, axis=-1)
        else:
            q, k, v = jnp.split(Linear(hidden_dim * 3,  bias=self.qkv_bias)(array), 3, axis=-1)
        q, k, v = map(lambda a: rearrange(a, "b n (h d) -> b h n d", h=self.num_heads), (q, k, v))
        
        if self.rotary_base:
            rope = Rotary(self.rotary_base, cls_index=self.cls_index, mask_cls=True)
            q = rope(q, pos_offset=0)
            k = rope(k, pos_offset=0)

        attn_weights = jnp.einsum("b h i d, b h j d -> b h i j", q, k) / jnp.sqrt(q.shape[-1])
        if apply_bias:
            attn_weights = attn_weights + apply_bias
        attn_weights = nn.softmax(attn_weights, axis=-1)

        attn_weights = nn.Dropout(rate=self.dropout_rate)(attn_weights, deterministic=self.deterministic)
        out = jnp.einsum("b h i j, b h j d -> b h i d", attn_weights, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        out = Linear(features)(out)
        return out
    
class ConvStemPatchify(nn.Module):
    hidden_dim: int
    strides: tuple = (2, 2, 2, 2)
    use_cls: bool = True

    @nn.compact
    def __call__(self, array : jnp.ndarray) -> jnp.ndarray:
        for i, s in enumerate(self.strides):
            array = nn.Conv(features=min(64*(2**i), self.hidden_dim),
                        kernel_size=(3,3), strides=(s,s), padding='SAME', use_bias=False)(array)
            array = nn.gelu(array)
            array = Norm("layer")(array)
        array = nn.Conv(self.hidden_dim, (1,1), (1,1), padding='SAME', use_bias=False)(array)

        B,H,W,C = array.shape
        tokens = array.reshape(B, H*W, C)

        if self.use_cls:
            cls = self.param('cls', nn.initializers.zeros, (1,1,C))
            tokens = jnp.concatenate([jnp.tile(cls, (B,1,1)), tokens], axis=1)
        return tokens
    
def _rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)

class Rotary(nn.Module):
    """Apply RoPE directly to a per-head tensor [B, H, N, D].
    Configuration:
      - base: rotary frequency base.
      - mask_cls: if True, do not rotate token at cls_index.
      - cls_index: index of the token to skip (default 0).
    """
    base: float = 10000.0
    mask_cls: bool = False
    cls_index: int = 0

    def setup(self):
        self._base = self.base
        self._mask_cls = self.mask_cls
        self._cls_index = self.cls_index

    @nn.compact
    def __call__(
        self,
        array: jnp.ndarray,
        *,
        pos_offset: int = 0,
        positions: jnp.ndarray | None = None,
        no_rotate_mask: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        assert array.ndim == 4, "Expected [B, H, N, D]"
        B, H, N, D = array.shape
        assert D % 2 == 0, "Head dim D must be even"

        dtype = array.dtype
        half = D // 2
        inv_freq = 1.0 / (self._base ** (jnp.arange(0, half, dtype=dtype) / half))

        t = (jnp.arange(pos_offset, pos_offset + N, dtype=dtype)
             if positions is None else positions.astype(dtype) + jnp.array(pos_offset, dtype=dtype))
        freqs = jnp.einsum("n,d->n d", t, inv_freq)
        ang = jnp.concatenate([freqs, freqs], axis=-1)
        cos = jnp.cos(ang)[None, None, :, :]
        sin = jnp.sin(ang)[None, None, :, :]

        if self._mask_cls:
            cls_mask = jnp.zeros((N,), dtype=bool).at[self._cls_index].set(True)
            eff_mask = cls_mask if no_rotate_mask is None else (no_rotate_mask | cls_mask)
        else:
            eff_mask = no_rotate_mask

        if eff_mask is not None:
            m = eff_mask[None, None, :, None]
            cos = jnp.where(m, jnp.ones_like(cos), cos)
            sin = jnp.where(m, jnp.zeros_like(sin), sin)

        return (array * cos) + (_rotate_half(array) * sin)