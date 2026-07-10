import torch
import torch.nn as nn

from .base import Encoder


class AttentionEncoder(Encoder):
    """Treats each ETF as a token and a macro block as an extra token, and
    lets self-attention learn how every other ETF (and macro conditions)
    influences each one's representation -- rather than flattening the whole
    basket into one undifferentiated vector (what MLPEncoder does).

    Keeps one contextualized embedding per ETF (drops the macro token after
    attention) so the policy still gets per-ETF resolution to size each
    position, instead of collapsing the whole basket into a single pooled
    vector.
    """

    def __init__(
        self,
        n_assets: int,
        per_asset_dim: int,
        macro_dim: int,
        latent_dim: int,
        d_model: int = 16,
        n_heads: int = 2,
    ):
        input_dim = n_assets * per_asset_dim + macro_dim
        super().__init__(input_dim, latent_dim)
        self.n_assets = n_assets
        self.per_asset_dim = per_asset_dim
        self.macro_dim = macro_dim
        self.d_model = d_model

        self.asset_proj = nn.Linear(per_asset_dim, d_model)
        self.macro_proj = nn.Linear(macro_dim, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(n_assets * d_model, latent_dim)
        self.decoder = nn.Linear(d_model, per_asset_dim)

    def _tokenize(self, x: torch.Tensor):
        batch = x.shape[0]
        split = self.n_assets * self.per_asset_dim
        assets = x[:, :split].view(batch, self.n_assets, self.per_asset_dim)
        macro = x[:, split:]
        return assets, macro

    def _asset_context(self, x: torch.Tensor) -> torch.Tensor:
        assets, macro = self._tokenize(x)
        asset_tokens = self.asset_proj(assets)
        macro_token = self.macro_proj(macro).unsqueeze(1)
        tokens = torch.cat([asset_tokens, macro_token], dim=1)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)
        return tokens[:, : self.n_assets, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self._asset_context(x)
        flat = context.reshape(x.shape[0], -1)
        return self.out_proj(flat)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        context = self._asset_context(x)
        return self.decoder(context).reshape(x.shape[0], -1)

    def reconstruction_target(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, : self.n_assets * self.per_asset_dim]
