"""InstantHMR Medium inference adapted from ComfyUI-InstantHMR (Apache-2.0).

The backbone and decoder retain the checkpoint's tensor names. This port uses
PyTorch layers directly and returns the geometry and camera predictions.
"""

import torch
from torch import nn
from torch.nn import functional as F

IMAGE_SIZE = 288
DECODER_WIDTH = 512
ATTENTION_HEADS = 8

# Input, internal and output channels, blocks, downsampling, light convs,
# kernel size and convolutions per block for HGNetV2-B4.
STAGES = (
    (48, 48, 128, 1, False, False, 3, 6),
    (128, 96, 512, 1, True, False, 3, 6),
    (512, 192, 1024, 3, True, True, 5, 6),
    (1024, 384, 2048, 1, True, True, 5, 6),
)


class ConvAct(nn.Module):
    """Convolution with optional ReLU and checkpoint-folded normalization."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        stride=1,
        *,
        groups=1,
        activate=True,
    ):
        super().__init__()

        # The checkpoint already folds batch normalization into these weights.
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel,
            stride,
            (kernel - 1) // 2,
            groups=groups,
        )
        self.activate = activate

    def forward(self, image):
        """Convolve the feature map and apply the configured activation."""
        result = self.conv(image)

        return F.relu(result) if self.activate else result


class LightConvAct(nn.Module):
    """Pointwise projection followed by a depthwise convolution and ReLU."""

    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()

        self.conv1 = ConvAct(in_channels, out_channels, 1, activate=False)
        self.conv2 = ConvAct(
            out_channels,
            out_channels,
            kernel,
            groups=out_channels,
        )

    def forward(self, image):
        """Project the channels and filter each output channel spatially."""

        return self.conv2(self.conv1(image))


class Stem(nn.Module):
    """HGNetV2 stem combining pooled and convolutional features."""

    def __init__(self):
        super().__init__()

        self.stem1 = ConvAct(3, 32, 3, 2)
        self.stem2a = ConvAct(32, 16, 2)
        self.stem2b = ConvAct(16, 32, 2)

        self.stem3 = ConvAct(64, 32, 3, 2)
        self.stem4 = ConvAct(32, 48, 1)
        self.pool = nn.MaxPool2d(2, 1, ceil_mode=True)

    def forward(self, image):
        """Reduce the input resolution by four and emit 48 channels."""
        features = F.pad(self.stem1(image), (0, 1, 0, 1))
        branch = self.stem2b(F.pad(self.stem2a(features), (0, 1, 0, 1)))

        return self.stem4(self.stem3(torch.cat((self.pool(features), branch), dim=1)))


class HGBlock(nn.Module):
    """
    Aggregate intermediate convolution outputs with an optional residual.

    Parameters
    ----------
    in_channels, channels, out_channels : int
        Channel counts for the block input, each internal layer, and the
        aggregated output.
    kernel, depth : int
        Spatial kernel size and number of internal layers.
    light : bool
        Use pointwise and depthwise layers when true.
    residual : bool
        Add the input to the aggregated output. Requires equal input and
        output channel counts.
    """

    def __init__(
        self,
        in_channels,
        channels,
        out_channels,
        kernel,
        depth,
        light,
        residual,
    ):
        super().__init__()

        layer = LightConvAct if light else ConvAct
        self.layers = nn.ModuleList(
            layer(in_channels if i == 0 else channels, channels, kernel)
            for i in range(depth)
        )

        self.aggregation = nn.Sequential(
            ConvAct(in_channels + depth * channels, out_channels // 2, 1),
            ConvAct(out_channels // 2, out_channels, 1),
        )
        self.residual = residual

    def forward(self, features):
        """Concatenate the input and each layer's output before aggregation."""
        outputs = [features]

        for layer in self.layers:
            outputs.append(layer(outputs[-1]))

        result = self.aggregation(torch.cat(outputs, dim=1))

        return result + features if self.residual else result


class HGStage(nn.Module):
    """Stack HG blocks with optional stride-two depthwise downsampling."""

    def __init__(
        self,
        in_channels,
        channels,
        out_channels,
        blocks,
        downsample,
        light,
        kernel,
        depth,
    ):
        super().__init__()

        self.downsample = (
            ConvAct(
                in_channels,
                in_channels,
                3,
                2,
                groups=in_channels,
                activate=False,
            )
            if downsample
            else nn.Identity()
        )

        self.blocks = nn.Sequential(*(
            HGBlock(
                in_channels if i == 0 else out_channels,
                channels,
                out_channels,
                kernel,
                depth,
                light,
                residual=i != 0,
            )
            for i in range(blocks)
        ))

    def forward(self, features):
        """Apply the stage's downsampling and feature blocks."""

        return self.blocks(self.downsample(features))


class HGNetV2(nn.Module):
    """HGNetV2-B4 feature backbone for the InstantHMR Medium checkpoint."""

    def __init__(self):
        super().__init__()

        self.stem = Stem()
        self.stages = nn.Sequential(*(HGStage(*stage) for stage in STAGES))

    def forward(self, image):
        """Encode 288-pixel crops as feature maps of shape ``(B, 2048, 9, 9)``."""

        return self.stages(self.stem(image))


class Attention(nn.Module):
    """Multi-head scaled dot-product attention with a shared key/value projection."""

    def __init__(self, width):
        super().__init__()

        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, width * 2)
        self.out = nn.Linear(width, width)

    def forward(self, query, context):
        """
        Attend from query tokens to context tokens.

        Parameters
        ----------
        query : torch.Tensor
            Query sequence of shape ``(B, Q, width)``.
        context : torch.Tensor
            Key/value sequence of shape ``(B, K, width)``. For self-attention,
            pass the query sequence as the context as well.

        Returns
        -------
        torch.Tensor
            Attended features of shape ``(B, Q, width)``.
        """
        key, value = self.kv(context).chunk(2, dim=-1)
        query = (
            self.q(query)
            .unflatten(-1, (ATTENTION_HEADS, -1))
            .transpose(1, 2)
        )
        key = key.unflatten(-1, (ATTENTION_HEADS, -1)).transpose(1, 2)
        value = value.unflatten(-1, (ATTENTION_HEADS, -1)).transpose(1, 2)

        attended = F.scaled_dot_product_attention(query, key, value)

        return self.out(attended.transpose(1, 2).flatten(2))


class DecoderLayer(nn.Module):
    """Pre-normalized transformer block with self- and cross-attention."""

    def __init__(self, width):
        super().__init__()

        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)

        self.self_attn = Attention(width)
        self.cross_attn = Attention(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
        )

    def forward(self, tokens, memory):
        """Update decoder tokens using each other and the image features."""
        normalized = self.norm1(tokens)
        tokens = tokens + self.self_attn(normalized, normalized)
        tokens = tokens + self.cross_attn(self.norm2(tokens), memory)

        return tokens + self.mlp(self.norm3(tokens))


class InstantHMRHead(nn.Module):
    """Regress 447 body coefficients as offsets from the checkpoint's estimate."""

    def __init__(self, width):
        super().__init__()

        self.proj = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 447),
        )

        self.register_buffer('init_estimate', torch.empty(1, 447))

    def forward(self, token):
        """Return continuous body coefficients with the initial estimate added."""

        return self.proj(token).float() + self.init_estimate


class InstantHMR(nn.Module):
    """
    Predict continuous MHR coefficients and camera translation for each crop.

    The decoder uses one body token and 70 keypoint tokens. Their interactions
    feed the body prediction, which is read from the first token. The runtime
    converts the resulting coefficients to the rig's native parameter layout.
    """

    def __init__(self):
        super().__init__()

        width = DECODER_WIDTH
        self.backbone = HGNetV2()
        self.feat_proj = nn.Linear(2048, width)
        self.register_buffer(
            'mem_pos_embed',
            torch.empty(1, (IMAGE_SIZE // 32) ** 2, width),
        )

        # The 70 keypoint tokens still participate in attention with the body token.
        self.register_buffer('query_embed', torch.empty(1, 71, width))
        self.cond_proj = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, width),
        )
        self.layers = nn.ModuleList(DecoderLayer(width) for _ in range(4))

        self.head_pose = InstantHMRHead(width)
        self.head_cam = nn.Linear(width, 3)

    def forward(self, image, cliff):
        """
        Infer body coefficients and camera translation for a batch of crops.

        Parameters
        ----------
        image : torch.Tensor
            ImageNet-normalized RGB crops of shape ``(B, 3, 288, 288)``.
        cliff : torch.Tensor
            Full-frame box conditioning of shape ``(B, 3)`` from
            ``prepare_crop``.

        Returns
        -------
        coefficients : torch.Tensor
            Continuous body parameters of shape ``(B, 447)``.
        camera_translation : torch.Tensor
            Camera-frame translation in metres with shape ``(B, 3)``.
        """
        memory = self.feat_proj(
            self.backbone(image).flatten(2).transpose(1, 2)
        )
        memory = memory + self.mem_pos_embed
        tokens = self.query_embed + self.cond_proj(cliff).unsqueeze(1)

        for layer in self.layers:
            tokens = layer(tokens, memory)

        body = tokens[:, 0]

        return self.head_pose(body), self.head_cam(body).float()
