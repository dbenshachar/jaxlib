from typing import Any
from modules import *

class VisionTransformer(nn.Module):
    hidden_dim : int = 1024
    num_heads : int = 16
    num_layers : int = 6
    num_patches : int = 8
    use_cls : bool = True
    pool : bool = False
    num_classes : int =  1
    activation : ModuleType = nn.gelu
    img_key : str = "array"
    return_key : str = "result"

    @nn.compact
    def __call__(self, batch : Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        image = batch[self.img_key]
        patch_size = (image.shape[-2]//self.num_patches, image.shape[-3]//self.num_patches)

        image = PatchEmbedding(self.hidden_dim, patch_size=patch_size, add_cls_token=self.use_cls)(image)
        for _ in range(self.num_layers):
            image = Norm("layer")(image)
            image = Attention(self.num_heads)(image) + image
            image = Norm("layer")(image)
            image = MLP(self.hidden_dim*4, activation=self.activation)(image) + image

        if self.pool:
            if self.use_cls:
                image = image.mean(axis=1)
            image = Linear(self.num_classes)(image)
        return {self.return_key : image}