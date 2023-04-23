import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """ 
    Image to Patch Embedding
    Parameters
    ----------
    img_size : int
        Size of the input image
    patch_size : int
        Size of the patches to be cut out of the input image
    in_chans : int 
        Number of input channels
    embed_dim : int
        Dimension of the embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # the output channels in the projection layer is the embedding dimension
        # we will need to flatten the spatial dimensions and transpose the output
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        '''
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor with shape (B, C, H, W)
        Returns
        -------
        x : torch.Tensor
            Patch embedding with shape (B, num_patches, embed_dim)
        '''
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        
        # projection to embedding dim (B, embed_dim, patches ** 0.5, patches ** 0.5)
        # then flatten the spatial dimensions (B, embed_dim, patches)
        # then transpose to (B, patches, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    """ 
    Attention Module
    Parameters
    ----------
    dim : int
        Dimension of the input
    num_heads : int
        Number of attention heads
    qkv_bias : bool
        If True, add a learnable bias to query, key, value projections.
    qk_scale : float or None
        Override default qk scale of head_dim ** -0.5 if set
    attn_drop : float
        Dropout ratio of the attention weight
    proj_drop : float
        Dropout ratio of the output
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of the full pre-attention projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        '''
        Parameters
        ----------
        x : torch.Tensor
            Input feature map with shape (B, N, C)
        Returns
        -------
        x : torch.Tensor
            Attention map with shape (B, N, C)
        '''
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # output shape: (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale # output shape: (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        print("Attention Map Shape: ", attn.shape)

        x = (attn @ v) # output shape: (B, num_heads, N, head_dim)
        x= x.transpose(1, 2) # output shape: (B, N, num_heads, head_dim)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class MLP(nn.Module):
    """ 
    MLP Module
    Parameters
    ----------
    in_features : int
        Number of input features
    hidden_features : int
        Number of features in the hidden layer
    out_features : int
        Number of output features
    act_layer : nn.Module
        Activation layer
    drop : float
        Dropout ratio
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        '''
        Parameters
        ----------
        x : torch.Tensor
            Input feature map with shape (B, N, C)
        Returns
        -------
        x : torch.Tensor
            MLP feature map with shape (B, N, C)
        '''
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# block input shape will be (B, N, C)
# block output shape will be (B, N, C)
class Block(nn.Module):
    """ 
    Transformer Block
    Parameters
    ----------
    dim : int
        Dimension of the input
    num_heads : int
        Number of attention heads
    mlp_ratio : float
        Ratio of mlp hidden dim to embedding dim
    qkv_bias : bool
        If True, add a learnable bias to query, key, value projections.
    qk_scale : float or None
        Override default qk scale of head_dim ** -0.5 if set
    drop : float
        Dropout ratio
    attn_drop : float
        Dropout ratio of the attention weight
    drop_path : float
        Stochastic depth rate
    act_layer : nn.Module
        Activation layer
    norm_layer : nn.Module
        Normalization layer
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # stochastic depth
        # skip connection with drop connect
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        '''
        Parameters
        ----------
        x : torch.Tensor
            Input feature map with shape (B, N, C)
        Returns
        -------
        x : torch.Tensor
            Transformer block feature map with shape (B, N, C)
        '''
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
# vision transformer pipeline consists of a patch embedding layer, a transformer encoder, and a classification head
# cls and pos tokens are added to the input image to form the input feature map
class VisionTransformer(nn.Module):
    """ 
    Vision Transformer
    Parameters
    ----------
    img_size : int
        Size of the input image
    patch_size : int
        Size of the patch
    in_chans : int
        Number of input image channels. (Default: 3)
    num_classes : int
        Number of classes for classification. (Default: 1000)
    embed_dim : int
        Embedding dimension. (Default: 768)
    depth : int
        Depth of the transformer. (Default: 12)
    num_heads : int
        Number of attention heads. (Default: 12)
    mlp_ratio : float
        Ratio of mlp hidden dim to embedding dim. (Default: 4)
    qkv_bias : bool
        If True, add a learnable bias to query, key, value projections. (Default: True)
    qk_scale : float or None
        Override default qk scale of head_dim ** -0.5 if set. (Default: None)
    representation_size : Optional[int]
        If None, use the model default, which is the embed_dim. If specified, use this value. (Default: None)
    drop_rate : float
        Dropout rate. (Default: 0.)
    attn_drop_rate : float
        Attention dropout rate. (Default: 0.)
    drop_path_rate : float
        Stochastic depth rate. (Default: 0.)
    norm_layer : nn.Module
        Normalization layer. (Default: nn.LayerNorm)
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                qk_scale=None, representation_size=None, drop_rate=0., attn_drop_rate=0., 
                drop_path_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.Blocks = nn.ModuleList(
                                    [
                                        Block(
                                                dim = embed_dim,
                                                num_heads = num_heads,
                                                mlp_ratio = mlp_ratio,
                                                qkv_bias = qkv_bias,
                                                drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                        )
                                        for _ in range(depth)
                                    ]
                                        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        '''
        Parameters
        ----------
        x : torch.Tensor
            Input image with shape (B, C, H, W)
        Returns
        -------
        x : torch.Tensor
            Classification logits with shape (B, num_classes)
        '''
        B = x.shape[0]
        x = self.patch_embed(x) # output shape (B, N, C)
        cls_tokens = self.cls_token.expand(B, -1, -1) # output shape (B, 1, C)
        x = torch.cat((cls_tokens, x), dim=1) # output shape (B, N+1, C)
        x = x + self.pos_embed # output shape (B, N+1, C)
        x = self.pos_drop(x) # output shape (B, N+1, C)
        for blk in self.Blocks:
            x = blk(x) # output shape (B, N+1, C)
        x = self.norm(x) # output shape (B, N+1, C)
        x = x[:, 0] # output shape (B, C)
        x = self.head(x) # output shape (B, num_classes)
        return x