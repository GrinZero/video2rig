"""The reader network: DINOv2 (Meta, Apache 2.0) sees the picture, a small decoder turns its patches into one heat
map per keypoint; a shallow branch on the picture itself keeps the fine edges the 14 px patches blur.

Input: a panel letterboxed into SIZE x SIZE (aspect kept, grey around it), ImageNet-normalised. Output: K heat maps at
stride STRIDE (cell i covers input pixels [i * STRIDE, (i + 1) * STRIDE)).

Speed on a laptop card (RTX 5070 Ti Laptop, warm): the first FROZEN transformer blocks run without gradients (they see
edges and fur, which our renders and Seedance's frames share), 448 px instead of DINOv2's own 518, a light decoder:
about 19 ms per picture and step instead of 80.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SIZE = 448                       # 32 x 32 patches of 14 px
STRIDE = 4                       # heat maps of 112 x 112
HEAT = SIZE // STRIDE
FROZEN = 4                       # transformer blocks that are not trained
CH = 96                          # decoder channels
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BACKBONE = 'facebook/dinov2-small'


def reader_autocast(dev='cuda'):
    """Use native BF16 where available; FP32 on T4 and CPU (no loss scaler required)."""
    from contextlib import nullcontext
    if str(dev).startswith('cuda') and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return nullcontext()


def block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Reader(nn.Module):
    """context > 0: the reader also takes `context` heat-map channels at its own output resolution: what it read in the
    frame before and the frame after (2 x K maps, a gaussian at every reading, its height the reading's confidence).
    A frame is then judged with the motion round it: a blurred leg read where its twin was is checked against where the
    leg was a moment before and after (read.py reads a clip twice: alone, then with the neighbours' first readings)."""

    def __init__(self, k, backbone=BACKBONE, pretrained=True, context=0, depth=False, skin=False, dense=0):
        """depth: a second map per keypoint, read where the keypoint is: how far it lies along the panel's line of
        sight, from the body's middle, in the creature's heights (stage R: one camera, the depth read).
        skin (with depth): four more maps, the skin map (dev/skin_map.py): for every cell, which place of the
        rest-pose model it shows (x, y, z from the rest pose's middle, in heights) and whether it is the creature.
        dense (the number of bones, with depth): the dense correspondence map (dense.py): for every cell which bone's
        skin shows there (dense + 1 classes, the last the background) and the place on that bone in the bone's own
        rest frame (3 numbers per bone, in heights; read on the bone named)."""
        super().__init__()
        from transformers import Dinov2Config, Dinov2Model
        self.vit = Dinov2Model.from_pretrained(backbone) if pretrained else Dinov2Model(Dinov2Config.from_pretrained(backbone))
        d = self.vit.config.hidden_size
        for p in self.vit.embeddings.parameters():
            p.requires_grad = False
        for blk in self.vit.encoder.layer[:FROZEN]:
            for p in blk.parameters():
                p.requires_grad = False
        self.fine = nn.Sequential(block(3, 32, 2), block(32, 48, 2))          # stride 4 from the picture
        self.proj = nn.Sequential(nn.Conv2d(d, CH, 1), nn.BatchNorm2d(CH), nn.ReLU(inplace=True))
        self.up1 = block(CH, CH)
        self.context = int(context)
        self.mix = nn.Sequential(block(CH + 48 + self.context, CH), block(CH, CH))
        self.head = nn.Conv2d(CH, k, 1)
        self.depth = bool(depth)
        if self.depth:
            self.dhead = nn.Sequential(block(CH, CH), nn.Conv2d(CH, k, 1))
        self.skin = bool(skin)
        assert not self.skin or self.depth, 'a skin reader is a depth reader too (its outputs: heat, depth, skin)'
        if self.skin:
            self.shead = nn.Sequential(block(CH, 128), block(128, 128), block(128, 128), nn.Conv2d(128, 4, 1))
        self.dense = int(dense)
        assert not self.dense or self.depth, 'a dense reader is a depth reader too (its outputs: heat, depth, parts, places)'
        if self.dense:
            self.phead = nn.Sequential(block(CH, 128), block(128, 128), nn.Conv2d(128, self.dense + 1, 1))
            self.lhead = nn.Sequential(block(CH, 128), block(128, 128), nn.Conv2d(128, 3 * self.dense, 1))
        self.k = k

    def forward(self, x, ctx=None):
        B, _, h, w = x.shape
        first = lambda o: o[0] if isinstance(o, tuple) else o
        with torch.no_grad():
            t = self.vit.embeddings(x)
            for blk in self.vit.encoder.layer[:FROZEN]:
                t = first(blk(t))
        for blk in self.vit.encoder.layer[FROZEN:]:
            t = first(blk(t))
        tok = self.vit.layernorm(t)[:, 1:]                                     # the class token dropped
        f = tok.transpose(1, 2).reshape(B, -1, h // 14, w // 14)
        f = self.up1(F.interpolate(self.proj(f), scale_factor=2, mode='bilinear', align_corners=False))
        size = (h // STRIDE, w // STRIDE)
        f = F.interpolate(f, size=size, mode='bilinear', align_corners=False)
        g = F.interpolate(self.fine(x), size=size, mode='bilinear', align_corners=False)
        parts = [f, g]
        if self.context:
            if ctx is None:                                                    # the first reading: no neighbours yet
                ctx = torch.zeros((B, self.context) + size, device=x.device, dtype=f.dtype)
            parts.append(F.interpolate(ctx.to(f.dtype), size=size, mode='bilinear', align_corners=False)
                         if tuple(ctx.shape[-2:]) != size else ctx.to(f.dtype))
        m = self.mix(torch.cat(parts, 1))
        if self.dense:
            return self.head(m), self.dhead(m), self.phead(m), self.lhead(m)
        if self.skin:
            return self.head(m), self.dhead(m), self.shead(m)
        return (self.head(m), self.dhead(m)) if self.depth else self.head(m)


def normalise(img):
    """uint8 (B, H, W, 3) BGR on the card -> float (B, 3, H, W) RGB, ImageNet-normalised."""
    x = img.flip(-1).permute(0, 3, 1, 2).float() / 255.0
    m = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def decode(heat):
    """Peak of every heat map, refined to a fraction of a cell (a parabola through the peak and its neighbours):
    (B, K, 2) input pixels and (B, K) the peak's height (the reader's confidence)."""
    B, K, h, w = heat.shape
    flat = heat.reshape(B, K, -1)
    conf, idx = flat.max(-1)
    y, x = (idx // w).float(), (idx % w).float()
    yi, xi = idx // w, idx % w

    def at(dy, dx):
        yy = (yi + dy).clamp(0, h - 1)
        xx = (xi + dx).clamp(0, w - 1)
        return flat.gather(-1, (yy * w + xx).unsqueeze(-1)).squeeze(-1)
    c = conf
    l, r, u, d = at(0, -1), at(0, 1), at(-1, 0), at(1, 0)
    dx = 0.5 * (r - l) / torch.where((2 * c - l - r).abs() > 1e-6, 2 * c - l - r, torch.ones_like(c))
    dy = 0.5 * (d - u) / torch.where((2 * c - u - d).abs() > 1e-6, 2 * c - u - d, torch.ones_like(c))
    x = x + dx.clamp(-.5, .5)
    y = y + dy.clamp(-.5, .5)
    return torch.stack([(x + .5) * STRIDE, (y + .5) * STRIDE], -1), conf


def sample_maps(maps, uv):
    """The value of map k at keypoint k's place: maps (B, K, h, w) at stride STRIDE, uv (B, K, 2) input pixels ->
    (B, K), bilinear (a place outside the picture reads the nearest edge)."""
    B, K, h, w = maps.shape
    gx = (uv[..., 0] / STRIDE) / w * 2 - 1                                 # cell i covers [i, i + 1) * STRIDE
    gy = (uv[..., 1] / STRIDE) / h * 2 - 1
    grid = torch.stack([gx, gy], -1).view(B * K, 1, 1, 2).to(maps.dtype)
    out = F.grid_sample(maps.reshape(B * K, 1, h, w), grid, mode='bilinear', padding_mode='border', align_corners=False)
    return out.view(B, K)


def letterbox(size_wh):
    """Scale and offset that fit a panel of size (w, h) into SIZE x SIZE: input = panel * s + (ox, oy)."""
    w, h = size_wh
    s = min(SIZE / w, SIZE / h)
    return s, ((SIZE - w * s) / 2, (SIZE - h * s) / 2)


def gaussians(uv, valid, sigma=1.5):
    """Target heat maps (B, K, HEAT, HEAT) for keypoints at input pixels uv (B, K, 2)."""
    B, K, _ = uv.shape
    c = uv / STRIDE - .5
    ax = torch.arange(HEAT, device=uv.device, dtype=uv.dtype)
    gx = torch.exp(-(ax.view(1, 1, -1) - c[..., 0:1]) ** 2 / (2 * sigma ** 2))
    gy = torch.exp(-(ax.view(1, 1, -1) - c[..., 1:2]) ** 2 / (2 * sigma ** 2))
    return gy.unsqueeze(-1) * gx.unsqueeze(-2) * valid[..., None, None]


def context_maps(uv_prev, conf_prev, uv_next, conf_next, sigma=2.0):
    """The neighbours' readings as context channels (B, 2K, HEAT, HEAT): a gaussian at every reading (input pixels
    (B, K, 2)), its height the reading's confidence (B, K), clipped to [0, 1]."""
    a = gaussians(uv_prev, conf_prev.clamp(0, 1), sigma=sigma)
    b = gaussians(uv_next, conf_next.clamp(0, 1), sigma=sigma)
    return torch.cat([a, b], 1)


def count_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def cosine(step, total, warm):
    if step < warm:
        return (step + 1) / warm
    t = min((step - warm) / max(total - warm, 1), 1.0)
    return .5 * (1 + math.cos(math.pi * t))
