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