import sys
sys.path.insert(0, '../')
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act

#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    # x * sqrt(x^2.mean)
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(nn.Module):
    def __init__(self,
                 in_features,                # 입력 특성(차원) 수
                 out_features,               # 출력 특성(차원) 수
                 bias            = True,     # 활성화 전에 바이어스(bias)를 추가할지 여부
                 activation      = 'linear', # 활성화 함수: 'relu', 'lrelu', 'linear' 등
                 lr_multiplier   = 1,        # 학습률 스케일 (learning rate multiplier)
                 bias_init       = 0,        # 바이어스 초기값
                 ):
        """
        Fully-connected layer (StyleGAN 구현부 등에서 사용).
        lr_multiplier와 bias_gain을 통해 가중치 초기화 및 학습률 효과를 조정.
        """
        super().__init__()

        # ----------------------------
        # 1) 가중치(weight) 파라미터
        # ----------------------------
        # weight shape: (out_features, in_features)
        #    => matmul 시 (batch, in_features) x (out_features, in_features)^T -> (batch, out_features)
        #
        # 초기화 시, (torch.randn(...) / lr_multiplier)은
        # lr_multiplier가 1보다 큰 경우, 가중치 스케일을 낮추는 효과.
        self.weight = nn.Parameter(
            torch.randn([out_features, in_features]) / lr_multiplier
        )

        # ----------------------------
        # 2) 바이어스(bias) 파라미터
        # ----------------------------
        # bias=True이면 (out_features,) 크기의 텐서를 생성.
        # bias_init 값으로 초기화.
        self.bias = nn.Parameter(
            torch.full([out_features], np.float32(bias_init))
        ) if bias else None

        # ----------------------------
        # 3) 내부 변수들
        # ----------------------------
        # activation: 'linear', 'relu', 'lrelu' 등
        self.activation = activation

        # weight_gain: 최종 forward 때 w에 곱해줄 스케일
        #    => lr_multiplier / sqrt(in_features)
        # bias_gain: 최종 forward 때 b에 곱해줄 스케일
        self.weight_gain = lr_multiplier / np.sqrt(in_features) # Xiavier initialization
        self.bias_gain = lr_multiplier

    def forward(self, x):
        """
        Forward 패스:
         - 입력 x에 대해 matmul(w^T)를 수행
         - 필요 시 bias와 활성화 함수를 적용
        """

        # w에 weight_gain을 곱해 실제 학습을 위한 스케일 조정
        w = self.weight * self.weight_gain

        # 바이어스가 있을 경우, bias_gain을 곱해 스케일 조정
        b = self.bias
        if b is not None and self.bias_gain != 1:
            b = b * self.bias_gain

        # ----------------------------
        # 1) 'linear'인 경우
        # ----------------------------
        #   -> PyTorch의 torch.addmm() 방식 또는
        #      x.matmul(w.t()) + b 형태로 구현
        if self.activation == 'linear' and b is not None:
            # x: (batch, in_features), w.t(): (in_features, out_features)
            # x.matmul(w.t()) => (batch, out_features)
            x = x.matmul(w.t())

            # out = x + b (방송(broadcast) 고려)
            #   => b는 (out_features, ) shape
            #   => x는 2차원 이상일 수도 있으므로 (x.ndim-1) dim에 맞춰 reshape
            out = x + b.reshape([-1 if i == x.ndim-1 else 1 for i in range(x.ndim)])

        else:
            # ----------------------------
            # 2) 그 외(예: 'relu', 'lrelu', ...)
            # ----------------------------
            #   -> bias_act.bias_act(x, b, act=...) 사용
            #   -> x.matmul(w.t()) 먼저 수행 후,
            #      bias + activation 동시 처리
            x = x.matmul(w.t())
            out = bias_act.bias_act(x, b, act=self.activation, dim=x.ndim-1) # 이거 custom 함수인데, 원하는 linear 함수 넣으면 통과시켜줌

        return out

#----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(nn.Module):
    def __init__(self,
                 in_channels,                    # Number of input channels.
                 out_channels,                   # Number of output channels.
                 kernel_size,                    # Width and height of the convolution kernel.
                 bias            = True,         # Apply additive bias before the activation function?
                 activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
                 up              = 1,            # Integer upsampling factor.
                 down            = 1,            # Integer downsampling factor.
                 resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
                 conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
                 trainable       = True,         # Update the weights of this layer during training?
                 ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter)) # trainable은 아닌데, state_dict 호출하면 불러오도록 buffer에 등록
        self.conv_clamp = conv_clamp
        self.padding = kernel_size // 2 
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2)) # element 개수가 in_channels * (kernel_size^2)여서 scaling해줌
        self.act_gain = bias_act.activation_funcs[activation].def_gain   # 이런게 있다는데 그렇게 중요한 거는 아님

        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size])
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        x = conv2d_resample.conv2d_resample(x=x, w=w, f=self.resample_filter, up=self.up, down=self.down,
                                            padding=self.padding)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        out = bias_act.bias_act(x, self.bias, act=self.activation, gain=act_gain, clamp=act_clamp)
        return out

#----------------------------------------------------------------------------

@persistence.persistent_class
class ModulatedConv2d(nn.Module): #Style을 배치/Out 채널별로 적용해주는 Conv class
    def __init__(self,
                 in_channels,                   # Number of input channels.
                 out_channels,                  # Number of output channels.
                 kernel_size,                   # Width and height of the convolution kernel.
                 style_dim,                     # dimension of the style code
                 demodulate=True,               # perfrom demodulation
                 up=1,                          # Integer upsampling factor.
                 down=1,                        # Integer downsampling factor.
                 resample_filter=[1,3,3,1],  # Low-pass filter to apply when resampling activations.
                 conv_clamp=None,               # Clamp the output to +-X, None = disable clamping.
                 ):
        super().__init__()
        self.demodulate = demodulate

        self.weight = torch.nn.Parameter(torch.randn([1, out_channels, in_channels, kernel_size, kernel_size]))
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.padding = self.kernel_size // 2
        self.up = up
        self.down = down
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.conv_clamp = conv_clamp

        self.affine = FullyConnectedLayer(style_dim, in_channels, bias_init=1)

    def forward(self, x, style):
        batch, in_channels, height, width = x.shape
        style = self.affine(style).view(batch, 1, in_channels, 1, 1) # style 벡터 projection
        weight = self.weight * self.weight_gain * style # (B, Out, In, K, K) 각 배치마다의 input 채널마다 style 곱해줌

        if self.demodulate: # demodulate(정규화)할 경우에
            decoefs = (weight.pow(2).sum(dim=[2, 3, 4]) + 1e-8).rsqrt() # (B, Out): L2 Norm
            weight = weight * decoefs.view(batch, self.out_channels, 1, 1, 1) # weight에 decoefs를 곱해 채널별/배치별로 정규화 

        weight = weight.view(batch * self.out_channels, in_channels, self.kernel_size, self.kernel_size) # (B * Out, In, K, K)
        x = x.view(1, batch * in_channels, height, width)
        
        # convolution 연산
        x = conv2d_resample.conv2d_resample(x=x, w=weight, f=self.resample_filter, up=self.up, down=self.down,
                                            padding=self.padding, groups=batch) 
        out = x.view(batch, self.out_channels, *x.shape[2:]) # (B, Out, K, K)

        return out

#----------------------------------------------------------------------------

@persistence.persistent_class
class StyleConv(torch.nn.Module): # ModulatedConv2d(스타일 반영 conv)에다가 noise를 추가해준 class
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        style_dim,                      # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        demodulate      = True,         # perform demodulation
    ):
        super().__init__()

        self.conv = ModulatedConv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    style_dim=style_dim,
                                    demodulate=demodulate,
                                    up=up,
                                    resample_filter=resample_filter,
                                    conv_clamp=conv_clamp)

        self.use_noise = use_noise
        self.resolution = resolution
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([])) # noise 얼마나 강하게 반영할 것인지는 학습시키자

        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.activation = activation
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.conv_clamp = conv_clamp

    def forward(self, x, style, noise_mode='random', gain=1):
        x = self.conv(x, style)

        assert noise_mode in ['random', 'const', 'none']

        if self.use_noise:
            if noise_mode == 'random':
                xh, xw = x.size()[-2:]
                noise = torch.randn([x.shape[0], 1, xh, xw], device=x.device) \
                        * self.noise_strength
            if noise_mode == 'const':
                noise = self.noise_const * self.noise_strength
            x = x + noise

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        out = bias_act.bias_act(x, self.bias, act=self.activation, gain=act_gain, clamp=act_clamp)

        return out

#----------------------------------------------------------------------------

@persistence.persistent_class
class ToRGB(torch.nn.Module): # 1x1 kernel을 사용하는 ModulatedConv2d (+ Residual connection) 적용해서 RGB feature로 바꾸는 class
    def __init__(self,
                 in_channels,
                 out_channels,
                 style_dim,
                 kernel_size=1,
                 resample_filter=[1,3,3,1],
                 conv_clamp=None,
                 demodulate=False):
        super().__init__()

        self.conv = ModulatedConv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    style_dim=style_dim,
                                    demodulate=demodulate,
                                    resample_filter=resample_filter,
                                    conv_clamp=conv_clamp)
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.conv_clamp = conv_clamp

    def forward(self, x, style, skip=None):
        x = self.conv(x, style)
        out = bias_act.bias_act(x, self.bias, clamp=self.conv_clamp)

        if skip is not None:
            if skip.shape != out.shape:
                skip = upfirdn2d.upsample2d(skip, self.resample_filter)
            out = out + skip

        return out

#----------------------------------------------------------------------------

@misc.profiled_function
def get_style_code(a, b): # 그냥 concat
    return torch.cat([a, b], dim=1)

#----------------------------------------------------------------------------

@persistence.persistent_class
class DecBlockFirst(nn.Module): # Decoder 부분에 처음 block-> fc로 맵핑하고  Encoder Feature 추가 (U-Net), modulated conv 통과, rgb로 mapping
    def __init__(self, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):
        super().__init__()
        self.fc = FullyConnectedLayer(in_features=in_channels*2,
                                      out_features=in_channels*4**2,
                                      activation=activation)
        self.conv = StyleConv(in_channels=in_channels,
                              out_channels=out_channels,
                              style_dim=style_dim,
                              resolution=4,
                              kernel_size=3,
                              use_noise=use_noise,
                              activation=activation,
                              demodulate=demodulate,
                              )
        self.toRGB = ToRGB(in_channels=out_channels,
                           out_channels=img_channels,
                           style_dim=style_dim,
                           kernel_size=1,
                           demodulate=False,
                           )

    def forward(self, x, ws, gs, E_features, noise_mode='random'):
        x = self.fc(x).view(x.shape[0], -1, 4, 4)
        x = x + E_features[2]
        style = get_style_code(ws[:, 0], gs) # 특정 레이어(?)의 스타일 벡터 w와 global style 벡터 g를 concat
        x = self.conv(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, 1], gs)
        img = self.toRGB(x, style, skip=None)

        return x, img


@persistence.persistent_class
class DecBlockFirstV2(nn.Module): #맨처음에 FC 대신에 conv사용
    def __init__(self, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):
        super().__init__()
        self.conv0 = Conv2dLayer(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=3,
                                activation=activation,
                                )
        self.conv1 = StyleConv(in_channels=in_channels,
                              out_channels=out_channels,
                              style_dim=style_dim,
                              resolution=4,
                              kernel_size=3,
                              use_noise=use_noise,
                              activation=activation,
                              demodulate=demodulate,
                              )
        self.toRGB = ToRGB(in_channels=out_channels,
                           out_channels=img_channels,
                           style_dim=style_dim,
                           kernel_size=1,
                           demodulate=False,
                           )

    def forward(self, x, ws, gs, E_features, noise_mode='random'):
        # x = self.fc(x).view(x.shape[0], -1, 4, 4)
        x = self.conv0(x)
        x = x + E_features[2]
        style = get_style_code(ws[:, 0], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, 1], gs)
        img = self.toRGB(x, style, skip=None)

        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class DecBlock(nn.Module): # 첫 Conv가 StyleConv로 바뀌고, style indexing 방법만 조금 달라짐
    def __init__(self, res, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):  # res = 2, ..., resolution_log2
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(in_channels=in_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               up=2,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.conv1 = StyleConv(in_channels=out_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.toRGB = ToRGB(in_channels=out_channels,
                           out_channels=img_channels,
                           style_dim=style_dim,
                           kernel_size=1,
                           demodulate=False,
                           )

    def forward(self, x, img, ws, gs, E_features, noise_mode='random'):
        style = get_style_code(ws[:, self.res * 2 - 5], gs)
        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + E_features[self.res]
        style = get_style_code(ws[:, self.res * 2 - 4], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, self.res * 2 - 3], gs)
        img = self.toRGB(x, style, skip=img)

        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNet(torch.nn.Module): # Generator랑 Discriminator에 쓰이는 class
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim

        # 각 단계마다의 feature dimension
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim] # [z_dim + embed_feature, layer_featueres, layer_featueres, ..., w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'): # 각 단계 프로파일링(CPU & GPU 사용량 등을 확인)을 위한 함수 (https://jh-bk.tistory.com/20)
            if self.z_dim > 0:
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y # concat input & conditioning label

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta)) # torch.lerp(end, weight): interpolation

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        #Truncation Trick이라고 하는데, W를 W_{avg}에 가까워지도록 보장한다고 함 <- 안정적, but 다양성 감소
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)

        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class DisFromRGB(nn.Module): # Discriminator에 RGB feature를 줄 때 사용
    def __init__(self, in_channels, out_channels, activation):  # res = 2, ..., resolution_log2
        super().__init__()
        self.conv = Conv2dLayer(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                activation=activation,
                                )

    def forward(self, x):
        return self.conv(x)

#----------------------------------------------------------------------------

@persistence.persistent_class
class DisBlock(nn.Module): # Discriminator 블록
    def __init__(self, in_channels, out_channels, activation):  # res = 2, ..., resolution_log2
        super().__init__()
        self.conv0 = Conv2dLayer(in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=3,
                                 activation=activation,
                                 )
        self.conv1 = Conv2dLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 down=2,
                                 activation=activation,
                                 )
        self.skip = Conv2dLayer(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                down=2,
                                bias=False,
                             )

    def forward(self, x):
        skip = self.skip(x, gain=np.sqrt(0.5))
        x = self.conv0(x)
        x = self.conv1(x, gain=np.sqrt(0.5))
        out = skip + x

        return out

#----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module): # Mode Collapse 방지, 같은 배치 안에 속한 표본들간의 '표준편차'가 낮은 경우를 discriminator가 감지할 수 있도록 한다고 함(GPT)
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape

        # 그룹 수(mini batch) 정하기
        with misc.suppress_tracer_warnings():  # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size),
                          torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F  # 채널도 num_channels기준으로 쪼개기

        y = x.reshape(G, -1, F, c, H,
                      W)  # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)  # [GnFcHW] Subtract mean over group. [GnFcHW] - [nFcHW]
        y = y.square().mean(dim=0)  # [nFcHW]  Calc variance over group. sum((x - mean)^2) / # of x
        y = (y + 1e-8).sqrt()  # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])  # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)  # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)  # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)  # [NCHW]   Append to input as new channels.
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
                 c_dim,                        # Conditioning label (C) dimensionality.
                 img_resolution,               # Input resolution.
                 img_channels,                 # Number of input color channels.
                 channel_base       = 32768,    # Overall multiplier for the number of channels.
                 channel_max        = 512,      # Maximum number of channels in any layer.
                 channel_decay      = 1,
                 cmap_dim           = None,     # Dimensionality of mapped conditioning label, None = default.
                 activation         = 'lrelu',
                 mbstd_group_size   = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
                 mbstd_num_channels = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
                 ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        resolution_log2 = int(np.log2(img_resolution))
        assert img_resolution == 2 ** resolution_log2 and img_resolution >= 4 # minimum resolution 16x16
        self.resolution_log2 = resolution_log2

        def nf(stage):
            return np.clip(int(channel_base / 2 ** (stage * channel_decay)), 1, channel_max) # Stage가 늘어날 수록 return값이 줄어든다

        if cmap_dim == None:
            cmap_dim = nf(2)
        if c_dim == 0:
            cmap_dim = 0
        self.cmap_dim = cmap_dim

        if c_dim > 0:
            self.mapping = MappingNet(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None)

        Dis = [DisFromRGB(img_channels+1, nf(resolution_log2), activation)] #RGB scale을 projection하는 레이어 추가
        for res in range(resolution_log2, 2, -1):
            Dis.append(DisBlock(nf(res), nf(res-1), activation)) # Discriminator 블록을 resolution 단계만큼 추가

        if mbstd_num_channels > 0:
            Dis.append(MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels)) # hint of model collapse for Discriminator
        Dis.append(Conv2dLayer(nf(2) + mbstd_num_channels, nf(2), kernel_size=3, activation=activation)) #projetion layer
        self.Dis = nn.Sequential(*Dis)

        self.fc0 = FullyConnectedLayer(nf(2)*4**2, nf(2), activation=activation)
        self.fc1 = FullyConnectedLayer(nf(2), 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, images_in, masks_in, c):
        x = torch.cat([masks_in - 0.5, images_in], dim=1) # mask 평균을 0으로 해서 학습 쉽게, 원래 이미지도 concat해서 넣음
        x = self.Dis(x)
        x = self.fc1(self.fc0(x.flatten(start_dim=1))) # FFN Layer

        if self.c_dim > 0:
            cmap = self.mapping(None, c) # Discriminaor가 판별하는데에 도음을 주는 conditional vector를 구하는 과정이라고 생각하면됨

        if self.cmap_dim > 0:
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim)) # conditional vector를 반영하고 scaling하여 원래 x의 scaling 유지

        return x
