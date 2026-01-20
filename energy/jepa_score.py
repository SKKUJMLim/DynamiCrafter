import torch
from torch.autograd.functional import jacobian

'''
# Jacobian + SVD (analysis)
x = x.to(device) # x: (B,C,H,W) or (B,C,T,H,W)
score = jepa_score_exact(encoder, x, eps=1e-6, pool="mean")
print(score.shape, score)
'''

def _pool_tokens(out: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """
    out: (B, N, D) e.g., (B, 256, 1408)
    return: (B, D)
    """
    if out.dim() != 3:
        raise ValueError(f"Expected token output (B,N,D), got {out.shape}")
    if mode == "mean":
        return out.mean(dim=1)
    elif mode == "max":
        return out.max(dim=1).values
    else:
        raise ValueError(f"Unknown pool mode: {mode}")

def jepa_score_exact(
    encoder,
    x: torch.Tensor,
    eps: float = 1e-6,
    pool: str = "mean",
) -> torch.Tensor:
    """
    Exact-ish JEPA-SCORE implementation following the paper listing:
      J = jacobian(lambda x: model(x).sum(0), inputs=x)
      J = J.flatten(2).permute(1,0,2)
      svdvals = torch.linalg.svdvals(J)
      score = log(svdvals).sum(1)

    encoder(x) is assumed to return tokens (B,N,D). We pool to (B,D).
    Returns: (B,) tensor
    """
    if x.dim() not in (4, 5):
        raise ValueError(f"x must be (B,C,H,W) or (B,C,T,H,W). Got {x.shape}")

    # jacobian needs grad tracking on x
    x = x.detach().requires_grad_(True)

    def emb_fn(inp: torch.Tensor) -> torch.Tensor:
        out = encoder(inp)                # (B,N,D)
        emb = _pool_tokens(out, pool)     # (B,D)
        return emb

    # Following the paper: sum over batch dimension -> output shape (D,)
    # jacobian output shape: (D, *x.shape) == (D, B, C, H, W) or (D, B, C, T, H, W)
    J = jacobian(lambda inp: emb_fn(inp).sum(0), inputs=x)

    # Reshape to per-sample matrices: (B, D, input_dim)
    # J: (D, B, ...)
    with torch.inference_mode():
        # flatten everything after (D,B) into one axis
        J = J.flatten(start_dim=2)        # (D, B, input_dim)
        J = J.permute(1, 0, 2).contiguous()  # (B, D, input_dim)

        # SVD singular values per sample
        svdvals = torch.linalg.svdvals(J)     # (B, min(D, input_dim))

        # JEPA-SCORE
        score = svdvals.clamp_min(eps).log().sum(dim=1)  # (B,)

    return score