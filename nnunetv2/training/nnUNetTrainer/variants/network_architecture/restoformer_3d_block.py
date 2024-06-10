import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat

from einops import rearrange
from einops.layers.torch import Rearrange


def actFunc(act, *args, **kwargs):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU()
    elif act == 'relu6':
        return nn.ReLU6()
    elif act == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif act == 'prelu':
        return nn.PReLU()
    elif act == 'rrelu':
        return nn.RReLU(0.1, 0.3)
    elif act == 'selu':
        return nn.SELU()
    elif act == 'celu':
        return nn.CELU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError


class Concat(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.add_module('fn', fn)

    def forward(self, x):
        return cat([x, self.fn(x)], dim=1)


##########################################################################
## Reformer
def video_to_3d(x):
    #x = rearrange(x, 'b c t h w -> b (t h w) c')
    b,c,t,h,w = x.shape
    x = x.view(b,c,t*h*w).permute((0,2,1)).contiguous()

    return x

def video_to_4d(x):
    #x = rearrange(x, 'b c t h w -> b (t h w) c')
    b,c,t,h,w = x.shape
    x = x.permute((0,2,1,3,4)).view(b*t,c,h,w).contiguous()

    return x

def video4d_to_5d(x,t):
    #x = rearrange(x, 'b (t h w) c -> b c t h w',t=t,h=h,w=w)
    bt,c,h,w = x.shape
    x = x.view(-1,t,c,h,w).permute((0,2,1,3,4)).contiguous()

    return x

def video_to_5d(x,t,h,w):
    #x = rearrange(x, 'b (t h w) c -> b c t h w',t=t,h=h,w=w)
    b,_,c = x.shape
    x = x.permute((0,2,1)).contiguous().view(b,c,t,h,w)

    return x

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        t, h, w = x.shape[-3:]
        return video_to_5d(self.body(video_to_3d(x)), t, h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv3d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv3d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
# Attention
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim*3*num_heads, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim*3*num_heads, dim*3*num_heads, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv3d(dim*num_heads, dim, kernel_size=1, bias=bias)
    
    def forward(self, x):
        b,c,t,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) t h w -> b head c (t h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) t h w -> b head c (t h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) t h w -> b head c (t h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (t h w) -> b (head c) t h w', head=self.num_heads, t=t, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=True, LayerNorm_type='WithBias'):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn  = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn   = FeedForward(dim, ffn_expansion_factor, bias)
       
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
