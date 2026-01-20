import torch
from torch.autograd.functional import jacobian
from typing import Deque, List, Optional, Tuple, Union

'''
# Jacobian + SVD (analysis)
x = x.to(device) # x: (B,C,H,W) or (B,C,T,H,W)
score = jepa_score_exact(encoder, x, eps=1e-6, pool="mean")
print(score.shape, score)
'''


@torch.no_grad()
def load_vjepa2_encoder(
    device: Union[str, torch.device],
    repo: str = "facebookresearch/vjepa2",
    model_name: str = "vjepa2_vit_giant",
) -> torch.nn.Module:
    """
    Load V-JEPA2 encoder via torch.hub and move to device.
    Returns encoder in eval() mode.
    """

    # # ---------------- V-JEPA2 (HF) load ----------------
    # processor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_preprocessor")
    # loaded = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_giant")
    #
    # if isinstance(loaded, tuple):
    #     vjepa2_encoder = loaded[0]  # encoder만 사용
    # else:
    #     vjepa2_encoder = loaded
    #
    # vjepa2_encoder = vjepa2_encoder.to(t2v_pipeline.model.device).eval()
    # # ---------------------------------------------------


    loaded = torch.hub.load(repo, model_name)

    if isinstance(loaded, tuple):
        encoder = loaded[0]  # encoder만 사용
    else:
        encoder = loaded

    encoder = encoder.to(device).eval()

    return encoder


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
    print("jacobian...")
    J = jacobian(lambda inp: emb_fn(inp).sum(0), inputs=x)
    print("jacobian END...")

    # Reshape to per-sample matrices: (B, D, input_dim)
    # J: (D, B, ...)
    with torch.inference_mode():
        # flatten everything after (D,B) into one axis
        J = J.flatten(start_dim=2)        # (D, B, input_dim)
        J = J.permute(1, 0, 2).contiguous()  # (B, D, input_dim)

        # SVD singular values per sample
        print("SVD...")
        svdvals = torch.linalg.svdvals(J)     # (B, min(D, input_dim))
        print("SVD End...")

        # JEPA-SCORE
        score = svdvals.clamp_min(eps).log().sum(dim=1)  # (B,)


    return score



import torch
from torch.autograd.functional import jvp

def jepa_energy_jvp(
    encoder_fn,
    x: torch.Tensor,
    n_dir: int = 4,
    eps: float = 1e-6,
    pool: str = "mean",
):
    """
    encoder_fn(x): (B,N,D) token output
    x: (B,C,T,H,W) or (B,C,H,W)
    return: (B,) energy
    """
    x = x.detach()
    B = x.shape[0]

    def emb_fn(inp):
        out = encoder_fn(inp)      # (B,N,D)
        if out.dim() == 3:
            emb = out.mean(dim=1)  # (B,D)
        else:
            emb = out
        return emb

    energies = []

    for _ in range(n_dir):
        v = torch.randn_like(x)
        v = v / (v.norm() + eps)

        # JVP: returns (emb, Jv)
        _, Jv = jvp(emb_fn, (x,), (v,), create_graph=False)

        # energy per sample
        e = (Jv ** 2).sum(dim=-1)  # (B,)
        energies.append(e)

    energy = torch.stack(energies, dim=0).mean(dim=0)  # (B,)
    return energy

import torch

def jepa_energy_fd(
    encoder_fn,
    x: torch.Tensor,
    n_dir: int = 2,
    eps: float = 1e-3,
):
    """
    Finite-difference JVP-based JEPA energy
    encoder_fn(x): (B,N,D) or (B,D)
    x: (B,C,T,H,W)
    return: (B,)
    """
    B = x.shape[0]
    energies = []

    with torch.no_grad():  # forward only
        f0 = encoder_fn(x)
        if f0.dim() == 3:
            f0 = f0.mean(dim=1)  # (B,D)

        for _ in range(n_dir):
            v = torch.randn_like(x)
            v = v / (v.norm() + 1e-6)

            f1 = encoder_fn(x + eps * v)
            if f1.dim() == 3:
                f1 = f1.mean(dim=1)

            Jv = (f1 - f0) / eps      # (B,D)
            # e = (Jv ** 2).sum(dim=-1)
            e = (Jv ** 2).mean(dim=-1) # average squared sensitivity per embedding dimension
            energies.append(e)

    return torch.stack(energies).mean(dim=0)
