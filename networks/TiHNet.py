import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from torchvision.ops.misc import Permute
import deformable_conv_v3



#-------------------------------------------------------------------

# ConvNext backbone #
class CNN_Block:
    def __init__(self, in_channels, channel_scale: int = 4, layer_scale: float = 1e-6):
        super(CNN_Block, self).__init__()

        # nn.Conv2d input shape: (B, C, H, W)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size = 7, padding = 1, groups = in_channels),
            Permute([0, 2, 3, 1]),
            nn.LayerNorm(in_channels, eps = 1e-6),
            Permute([0, 3, 1, 2]),
            nn.Conv2d(in_channels, in_channels * channel_scale, kernel_size = 1),
            nn.GELU(),
            nn.Conv2d(in_channels * channel_scale, in_channels, kernel_size = 1)
        )

        self.layer_scale = nn.Parameter(torch.ones(1, in_channels, 1, 1) * layer_scale)


    def forward(self, x):
        residual = self.cnn(x) * self.layer_scale  #어떤 channel이 중요한지 layer_scale 곱셈

        out = x + residual
        return out


class MHA(nn.Module):
    """
    utilize CvT Attention Module
    original code url: https://github.com/leoxiaobin/CvT/blob/main/lib/models/cls_cvt.py

    """

    def __init__(self, in_channels, out_channels, head_num, qkv_bias = False, attn_drop = 0, proj_drop = 0,
                 kernel_size = 3, stride_kv = 1, stride_q = 1, padding_kv = 1, padding_q = 1, with_cls_token = False):
        
        """
        Args:

        in_channels:        # Number of input channel
        out_channels:       # Number of output channel
        head_num:           # Number of head when doing MHSA operations
        qkv_bias:           # Bias of CNN filters projecting feature map to Query, Key, and Value 
        attn_drop:          # Probability of dropout right before multiplying attention socres with Value matrix
        proj_drop:          # Probability of dropout of projection, which projecting input to Query, Key, and Value matrix
        kernel_size:        # Namely.
        stride_kv:          # Stride of CNN p   rojection filter when projecting input to Key and Value
        stride_q:           # Stride of CNN projection filter when projecting input to Query
        padding_kv:         # Padding of CNN projection when projecting input to Key and Value
        padding_q:          # Padding of CNN projection when projecting input to Query
        with_cls_token:     # Whether use cls token
        """

        super(MHA, self).__init__()

        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.out_channels = out_channels
        self.head_num = head_num
        self.with_cls_token = with_cls_token

        # Transformer 원 논문에 따라서 Scaling
        self.scale = torch.sqrt(torch.tensor(out_channels / head_num))

        # fixing method as 'dw_bn' (depth-wise BatchNormalization 사용하겠다는 뜻)
        self.method = "dw_bn"

        # Overlapping-Tokenization with Deformable Convolution v3
        self.dcn_proj_q = self._build_projection(in_channels, out_channels, kernel_size,\
                                                  padding_q, stride_q, self.method)
        
        self.dcn_proj_k = self._build_projection(in_channels, out_channels, kernel_size,\
                                                  padding_kv, stride_kv, self.method)
        
        self.dcn_proj_v = self._build_projection(in_channels, out_channels, kernel_size,\
                                                  padding_kv, stride_kv, self.method)
        

        # Query, Key, Value를 Hidden dimension으로 projection
        self.proj_q = nn.Linear(in_channels, out_channels, bias = qkv_bias)
        self.proj_k = nn.Linear(in_channels, out_channels, bias = qkv_bias)
        self.proj_v = nn.Linear(in_channels, out_channels, bias = qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        #
        self.proj = nn.Linear(in_channels, out_channels)
        self.proj_drop = nn.Dropout(proj_drop)



    def _build_projection(self, in_channels, out_channels, kernel_size, padding, stride, method = None):
        if method == 'dw_bn':

            # Rearrange method: 차원 바꿔주는 메소드 
            # -> 마치 (B, L, C)로 바꿔서 token처럼 사용 (L: Sequence Length)
            proj = nn.Sequential(
                deformable_conv_v3(in_channels, out_channels, kernel_size, padding = padding,\
                          stride = stride, bias = False),
                nn.BatchNorm2d(in_channels),
                Rearrange('B C H W -> B (H W) C')
            )

        elif method is not None:
            proj = nn.Sequential(
                deformable_conv_v3(in_channels, out_channels, kernel_size, padding = padding,\
                          stride = stride, bias = False),
                          Rearrange('B C H W -> B (H W) C')
            )
        else:
            raise ValueError(f'Unknown method- {method}. Please change Normalization method.')
        
        return proj

    def foward(self, x):
        # shape of x: (B, H * W, C)

        _, t, _ = x.size()
        h, w = int(np.sqrt(t-1)) if self.with_cls_token else int(np.sqrt(t))

        # 분류 문제 아니어서 CLS 토큰 사용 x
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, h*w], 1)
        
        q = self.dcn_proj_q(x)
        k = self.dcn_proj_k(x)
        v = self.dcn_proj_v(x)

        if self.with_cls_token:
            q = torch.cat((cls_token, q), dim = 1)
            k = torch.cat((cls_token, k), dim = 1)
            v = torch.cat((cls_token, v), dim = 1)

        
        q = rearrange(self.proj_q(q), 'B L (H D) -> B H L D') # H: Number of Head
        k = rearrange(self.proj_k(k), 'B L (H D) -> B H L D')
        v = rearrange(self.proj_v(v), 'B L (H D) -> B H L D')


        attn_score = (q @ k.transpose(-2, -1)) / self.scale # output shape: (B, H, L, L)
        attn_weight = self.attn_drop(torch.softmax(attn_score, dim = -1)) # output shape: (B, H, L, L)

        attention = attn_weight @ v # output shae: (B, H, L, D)
        attention = rearrange(attention, 'B H L D -> B L (H D)')

        attention = self.proj(attention)
        attention = self.proj_drop(attention)

        return attention
    

    # 아래는 파라미터 개수 구하기 위한 메소드
    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        flops = 0

        _, T, C = input.shape
        H = W = int(np.sqrt(T-1)) if module.with_cls_token else int(np.sqrt(T))

        H_Q = H / module.stride_q
        W_Q = H / module.stride_q
        T_Q = H_Q * W_Q + 1 if module.with_cls_token else H_Q * W_Q

        H_KV = H / module.stride_kv
        W_KV = W / module.stride_kv
        T_KV = H_KV * W_KV + 1 if module.with_cls_token else H_KV * W_KV

        # C = module.dim
        # S = T
        # Scaled-dot-product macs
        # [B x T x C] x [B x C x T] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        flops += T_Q * T_KV * module.out_channels
        # [B x T x S] x [B x S x C] --> [B x T x C]
        flops += T_Q * module.out_channels * T_KV

        if (hasattr(module, 'conv_proj_q')and hasattr(module.conv_proj_q, 'conv')):

            params = sum([p.numel() for p in module.conv_proj_q.conv.parameters()])
            flops += params * H_Q * W_Q

        if (hasattr(module, 'conv_proj_k') and hasattr(module.conv_proj_k, 'conv')):

            params = sum([p.numel() for p in module.conv_proj_k.conv.parameters()])
            flops += params * H_KV * W_KV

        if (hasattr(module, 'conv_proj_v') and hasattr(module.conv_proj_v, 'conv')):

            params = sum([p.numel()for p in module.conv_proj_v.conv.parameters()])
            flops += params * H_KV * W_KV

        params = sum([p.numel() for p in module.proj_q.parameters()])
        flops += params * T_Q
        params = sum([p.numel() for p in module.proj_k.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj_v.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj.parameters()])
        flops += params * T

        module.__flops__ += flops



#-------------------------------------------------------------------
        
class FFN(nn.Module):
    def __init__(self, d_model, d_ff, drop_p = 0.):
        """
        Args:

        d_model             # dimension of model
        d_ff                # dimension of hidden layer in FFN
        drop_p              # probability of dropout between FFN
        """     

        super(FFN, self).__init__()

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        # shape of x: (B, L, D)

        x = self.ffn(x)
        return x
    

#-------------------------------------------------------------------
class Transformer_Block(nn.Module):
    def __init__(self, in_channels, out_channels, head_num, ffn_ratio = 4., qkv_bias = False,
                 drop_p = 0., attn_drop_p = 0., with_cls_token = False):
        """
        Args:

        in_channels             # Number of input channel
        out_channels            # Number of output channel (d_model)
        head_num                # Number of head in MHSA
        ffn_ratio               # Sacling factor of dimension in FFN's hidden layer
        qkv_bias                # Whether using bias in Query, Key, and Value projection
        drop_p                  # probability of dropout in projection 
        attn_drop_p             # probability of dropout in QKV projection in MHA
        with_cls_token          # Wthether using cls_token
        """

        super(Transformer_Block, self).__init__()

        self.with_cls_token = with_cls_token

        self.norm1 = nn.LayerNorm(in_channels, eps = 1e-6)
        self.attn = MHA(in_channels, out_channels, head_num, qkv_bias, attn_drop_p, drop_p)
        self.norm2 = nn.LayerNorm(out_channels, eps = 1e-6)
        self.ffn = FFN(out_channels, int(out_channels * ffn_ratio), drop_p = drop_p)

    def forward(self, x):
        # shape of x: (B, L, D)

       res = self.norm(x)
       res = self.attn(res)
       x = x + res # output shape: (B, L, D)

       res = self.norm2(x)
       res = self.ffn(res)
       x = x + res

       return x
    


