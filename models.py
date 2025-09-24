from typing import Any
from modules import *

class VisionTransformer(nn.Module):
    hidden_dim : int = 1024
    num_heads : int = 16
    num_layers : int = 6
    num_patches : int = 8
    strides : Tuple[int, int, int, int] = (2, 2, 2, 2)
    use_cls : bool = True
    pool : bool = False
    num_classes : int =  1
    activation : ModuleType = GeGLU()
    img_key : str = "array"
    return_key : str = "result"
    dropout_rate : float = 0.1
    qkv_bias : bool = True
    deterministic : bool = False

    @nn.compact
    def __call__(self, batch : Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        image = batch[self.img_key]

        patch_size = (image.shape[-2]//self.num_patches), (image.shape[-3]//self.num_patches)
        image = PatchEmbedding(self.hidden_dim, patch_size=patch_size)(image)
        image = Transformer(self.hidden_dim, self.num_heads, self.num_layers, self.activation, self.img_key, self.return_key, self.dropout_rate, deterministic=self.deterministic)({self.img_key : image})[self.return_key]

        if self.pool:
            if self.use_cls:
                image = image[:, 0]
            else:
                image = image.mean(axis=1)
            image = Linear(self.num_classes if self.pool else image.shape[-1])(image)
        image = Norm("layer")(image)
        return {self.return_key : image}
    
class SequenceTransformer(nn.Module):
    num_embeddings : int
    hidden_dim : int = 1024
    num_heads : int = 16
    num_layers : int = 6
    activation : ModuleType = GeGLU()
    arr_key : str = "array"
    return_key : str = "result"
    dropout_rate : float = 0.1
    qkv_bias : bool = True
    deterministic : bool = False

    @nn.compact
    def __call__(self, batch : Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        array = batch[self.arr_key]
        array = nn.Embed(self.num_embeddings, self.hidden_dim)(array)
        image = Transformer(self.hidden_dim, self.num_heads, self.num_layers, self.activation, self.arr_key, self.return_key, self.dropout_rate, deterministic=self.deterministic)({self.arr_key : array})[self.return_key]

        image = Linear(self.num_embeddings)(array)
        image = Norm("layer")(image)
        return {self.return_key : image}

class Transformer(nn.Module):
    hidden_dim : int = 1024
    num_heads : int = 16
    num_layers : int = 6
    activation : ModuleType = GeGLU()
    arr_key : str = "array"
    return_key : str = "result"
    dropout_rate : float = 0.1
    qkv_bias : bool = True
    deterministic : bool = False

    @nn.compact
    def __call__(self, batch : Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        array = batch[self.arr_key]

        for depth in range(self.num_layers):
            drop_path_rate = max(0.2, depth * 0.01)

            res = Norm("layer")(array)
            res = Attention(self.num_heads, qkv_bias=self.qkv_bias, dropout_rate=self.dropout_rate)(res, deterministic=self.deterministic)
            res = LayerScale()(res)
            res = DropPath(drop_path_rate)(res, deterministic=self.deterministic)
            array += res

            res = Norm("layer")(array)
            res = MLP(self.hidden_dim*4, activation=self.activation, dropout_rate=self.dropout_rate)(res, deterministic=self.deterministic)
            res = LayerScale()(res)
            res = DropPath(drop_path_rate)(res, deterministic=self.deterministic)
            array += res
        return {self.return_key : array}