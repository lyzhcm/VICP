"""DINOv2 ViT implemented locally, loads weights from HuggingFace (bypasses torch.hub/GitHub)."""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Fused QKV attention matching torch.hub DINOv2 interface."""
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=14, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class DinoVisionTransformer(nn.Module):
    """DINOv2 ViT with torch.hub-compatible API."""
    def __init__(self, img_size=224, patch_size=14, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = 1
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.patch_size = patch_size

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return x

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((self.pos_embed[:, :1], patch_pos_embed), dim=1)

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.prepare_tokens_with_masks(x)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        return {"x_norm_clstoken": x_norm[:, 0], "x_prenorm": x}


def load_dinov2(model_name="dinov2_vitb14"):
    """Load DINOv2 - try torch.hub first, fall back to local HF loader."""
    # Try torch.hub first
    try:
        encoder = torch.hub.load("facebookresearch/dinov2", model_name).eval()
        print(f"[*] Loaded DINOv2 from torch.hub")
        return encoder
    except Exception as e:
        print(f"[!] torch.hub unavailable ({e}), loading from HuggingFace...")

    from huggingface_hub import hf_hub_download

    hf_model = "facebook/dinov2-base"

    if model_name == "dinov2_vits14":
        embed_dim, depth, num_heads = 384, 12, 6
        hf_model = "facebook/dinov2-small"
    elif model_name == "dinov2_vitl14":
        embed_dim, depth, num_heads = 1024, 24, 16
        hf_model = "facebook/dinov2-large"
    elif model_name == "dinov2_vitg14":
        embed_dim, depth, num_heads = 1536, 40, 24
        hf_model = "facebook/dinov2-giant"
    else:  # vitb14
        embed_dim, depth, num_heads = 768, 12, 12

    print(f"[*] Creating DINOv2 ViT: dim={embed_dim}, depth={depth}, heads={num_heads}")

    model = DinoVisionTransformer(
        img_size=224, patch_size=14, in_chans=3,
        embed_dim=embed_dim, depth=depth,
        num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True,
    )

    print(f"[*] Downloading weights from {hf_model} ...")
    state_dict = {}
    try:
        bin_path = hf_hub_download(repo_id=hf_model, filename="pytorch_model.bin")
    except Exception:
        # HF might have safetensors only
        from safetensors.torch import load_file
        bin_path = hf_hub_download(repo_id=hf_model, filename="model.safetensors")
        hf_state = load_file(bin_path)
        state_dict = hf_state

    if not state_dict:
        hf_state = torch.load(bin_path, map_location="cpu")

    # Map HF keys to our keys
    remap = {}
    for k, v in hf_state.items():
        if k.startswith("dinov2."):
            k = k[7:]
        remap[k] = v

    new_state = {}
    # pos_embed
    new_state["pos_embed"] = remap.get("embeddings.position_embeddings", remap.get("pos_embed"))
    new_state["cls_token"] = remap.get("embeddings.cls_token", remap.get("cls_token"))

    # patch_embed.proj
    new_state["patch_embed.proj.weight"] = remap["embeddings.patch_embeddings.projection.weight"]
    new_state["patch_embed.proj.bias"] = remap.get("embeddings.patch_embeddings.projection.bias", None)
    if new_state["patch_embed.proj.bias"] is None:
        new_state.pop("patch_embed.proj.bias")

    # norm
    new_state["norm.weight"] = remap["layernorm.weight"]
    new_state["norm.bias"] = remap["layernorm.bias"]

    # blocks
    for i in range(depth):
        prefix = f"encoder.layer.{i}."
        new_prefix = f"blocks.{i}."
        new_state[new_prefix + "norm1.weight"] = remap[prefix + "norm1.weight"]
        new_state[new_prefix + "norm1.bias"] = remap[prefix + "norm1.bias"]
        new_state[new_prefix + "norm2.weight"] = remap[prefix + "norm2.weight"]
        new_state[new_prefix + "norm2.bias"] = remap[prefix + "norm2.bias"]

        new_state[new_prefix + "mlp.fc1.weight"] = remap[prefix + "mlp.fc1.weight"]
        new_state[new_prefix + "mlp.fc1.bias"] = remap[prefix + "mlp.fc1.bias"]
        new_state[new_prefix + "mlp.fc2.weight"] = remap[prefix + "mlp.fc2.weight"]
        new_state[new_prefix + "mlp.fc2.bias"] = remap[prefix + "mlp.fc2.bias"]

        attn_prefix = prefix + "attention.attention."
        our_attn = new_prefix + "attn."
        q_w = remap[attn_prefix + "query.weight"]
        k_w = remap[attn_prefix + "key.weight"]
        v_w = remap[attn_prefix + "value.weight"]
        q_b = remap.get(attn_prefix + "query.bias", None)
        k_b = remap.get(attn_prefix + "key.bias", None)
        v_b = remap.get(attn_prefix + "value.bias", None)

        new_state[our_attn + "qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)
        if q_b is not None:
            new_state[our_attn + "qkv.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

        new_state[our_attn + "proj.weight"] = remap[prefix + "output.dense.weight"]
        new_state[our_attn + "proj.bias"] = remap[prefix + "output.dense.bias"]

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[!] Missing keys: {missing[:3]}...")
    if unexpected:
        print(f"[!] Unexpected keys: {unexpected[:3]}...")

    print(f"[+] DINOv2 loaded successfully")
    return model
