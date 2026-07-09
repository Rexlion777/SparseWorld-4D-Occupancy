from __future__ import annotations

import torch

from conftest import load_stage_module


def test_transformer_forward_shapes() -> None:
    mod = load_stage_module()
    model = mod.CrossAttentionCandidateTransformer(query_dim=len(mod.H_QUERY_FEATURES), token_dim=mod.TOKEN_DIM, d_model=32, num_heads=4, num_layers=2)
    batch = 7
    query = torch.randn(batch, len(mod.H_QUERY_FEATURES))
    tokens = torch.randn(batch, mod.MAX_TOKENS, mod.TOKEN_DIM)
    token_types = torch.arange(mod.MAX_TOKENS).unsqueeze(0).expand(batch, mod.MAX_TOKENS)
    token_valid = torch.ones(batch, mod.MAX_TOKENS, dtype=torch.bool)
    logits, uncertainty, attn = model(query, tokens, token_types, token_valid, need_weights=True)
    assert logits.shape == (batch,)
    assert uncertainty.shape == (batch,)
    assert attn is not None
    assert attn.shape[0] == batch
