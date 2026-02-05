import torch
from torch.autograd.functional import jacobian
from typing import Deque, List, Optional, Tuple, Union
from torch.autograd.functional import jvp
from typing import Callable, Literal, Tuple


print("[LOADED] energy.jepa_score from:", __file__)

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


# ----------------------------------------
# Pooling: encoder output -> embedding (D)
# ----------------------------------------
# @torch.no_grad()
# def _pool_tokens_if_needed(out: torch.Tensor, pool: str = "mean") -> torch.Tensor:
#     # out: (N,D) or (D,)
#     if out.dim() == 2:
#         if pool == "mean":
#             return out.mean(dim=0)
#         elif pool == "max":
#             return out.max(dim=0).values
#         else:
#             raise ValueError(f"Unknown pool: {pool}")
#     elif out.dim() == 1:
#         return out
#     else:
#         raise ValueError(f"Expected (N,D) or (D,), got {out.shape}")
@torch.no_grad()
def _pool_tokens_if_needed(out: torch.Tensor, pool: str = "mean") -> torch.Tensor:
    """
    Convert encoder output to (B, D).

    Supported shapes:
      - (B, D)
      - (D,)
      - (B, N, D)  : token sequence
      - (B, D, N)  : token sequence (rare; supported)
    """
    if not torch.is_tensor(out):
        raise ValueError(f"Expected torch.Tensor, got {type(out)}")

    # (D,) -> (1,D)
    if out.dim() == 1:
        return out.unsqueeze(0)

    # (B,D) ok
    if out.dim() == 2:
        return out

    # (B,N,D) or (B,D,N)
    if out.dim() == 3:
        B, A, C = out.shape

        if pool in ["mean", "avg"]:
            # Heuristic: assume last dim is embedding dim (D) when it's not huge token count.
            # In your case [1, 2048, 1408] => treat as (B,N,D) and pool over N.
            # If it is actually (B,D,N), this still works if you switch to pool="mean_tokens_last".
            return out.mean(dim=1)

        elif pool in ["mean_last", "mean_tokens_last"]:
            # If output is (B,D,N), pool over last dim
            return out.mean(dim=2)

        elif pool in ["first", "cls"]:
            # take first token: (B,D) if (B,N,D)
            return out[:, 0, :]

        elif pool in ["last"]:
            # take last token: (B,D) if (B,N,D)
            return out[:, -1, :]

        else:
            raise ValueError(f"Unknown pool='{pool}'. Use mean/mean_last/cls/last.")

    raise ValueError(f"Expected (B,D) or (D,) or (B,N,D)/(B,D,N), got {out.shape}")

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

def jepa_energy_fd(encoder_fn, x, n_dir=2, eps=1e-3):


    print("[HUTCH] jepa_energy_fd CALLED", flush=True)

    B = x.shape[0]
    energies = []

    with torch.no_grad():
        f0 = encoder_fn(x)
        if f0.dim() == 3:
            f0 = f0.mean(dim=1)

        for _ in range(n_dir):
            v = torch.randn_like(x)
            v = v / (
                v.flatten(1).norm(dim=1, keepdim=True)
                 .view(B, *([1] * (x.dim() - 1)))
                 .clamp_min(1e-6)
            )

            f1 = encoder_fn(x + eps * v)
            if f1.dim() == 3:
                f1 = f1.mean(dim=1)

            Jv = (f1 - f0) / eps
            e = (Jv ** 2).mean(dim=-1)
            energies.append(e)

    return torch.stack(energies).mean(dim=0)


@torch.no_grad()
def fd_hutchinson_trace_jtj(
    encoder_fn,
    x: torch.Tensor,
    n_samples: int = 4,
    noise: str = "rademacher",   # <-- 추가
    pool: str = "mean",
    normalize_r: bool = False,
    eps: float = 1e-8,           # <-- 호환용(정규화 안정성)
    eps_fd: float = 1e-3,        # <-- FD step
) -> torch.Tensor:
    """
    FD-Hutchinson estimator for Tr(J^T J)
    Jr ≈ (f(x + eps_fd * r) - f(x)) / eps_fd
    """

    print("[HUTCH] fd_hutchinson_trace_jtj CALLED", flush=True)

    def f(inp):
        out = encoder_fn(inp)
        emb = _pool_tokens_if_needed(out, pool)  # (B,D)
        return emb

    f0 = f(x)  # (B,D)
    estimates = []

    for _ in range(n_samples):
        if noise == "rademacher":
            r = _sample_rademacher_like(x)
        elif noise == "gaussian":
            r = torch.randn_like(x)
        else:
            raise ValueError(f"Unknown noise: {noise}")

        if normalize_r:
            r = r / (
                r.flatten(1).norm(dim=1, keepdim=True)
                .view(x.shape[0], *([1] * (x.dim() - 1)))
                .clamp_min(eps)
            )

        f1 = f(x + eps_fd * r)
        Jr = (f1 - f0) / eps_fd  # (B,D)
        # e = (Jr ** 2).sum(dim=-1)  # (B,)
        e = (Jr ** 2).mean(-1)
        estimates.append(e)

    return torch.stack(estimates, dim=0).mean(dim=0)




def _sample_rademacher_like(x: torch.Tensor) -> torch.Tensor:
    # +/-1 with equal prob
    return torch.empty_like(x).bernoulli_(0.5).mul_(2).sub_(1)


def hutchinson_trace_jtj(
    encoder_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    n_samples: int = 4,
    noise: Literal["rademacher", "gaussian"] = "rademacher",
    pool: str = "mean",
    normalize_r: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Hutchinson estimator for Tr(J^T J) = E_r[ ||J r||^2 ],
    where J = d f(x) / d x,  f(x) is the pooled embedding (B,D).

    Args:
      encoder_fn: function mapping x -> tokens (B,N,D) or embedding (B,D)
      x: (B,C,H,W) or (B,C,T,H,W)
      n_samples: number of Hutchinson probe vectors
      noise: 'rademacher' (±1) or 'gaussian' (N(0,1))
      pool: how to pool tokens if encoder_fn outputs (B,N,D)
      normalize_r: if True, normalize each sample's r to unit norm (NOT unbiased for trace;
                   can reduce variance but changes the quantity)
      eps: numerical stability for norm

    Returns:
      trace_est: (B,) tensor, per-sample estimate of Tr(J^T J)
    """

    print("[HUTCH] hutchinson_trace_jtj CALLED", flush=True)


    if x.dim() not in (4, 5):
        raise ValueError(f"x must be (B,C,H,W) or (B,C,T,H,W). Got {x.shape}")

    # We need autograd to compute JVP through encoder_fn
    x_req = x.detach().requires_grad_(True)

    def f(inp: torch.Tensor) -> torch.Tensor:
        out = encoder_fn(inp)                  # (B,N,D) or (B,D)
        emb = _pool_tokens_if_needed(out, pool)  # (B,D)
        print("encoder out:", out.shape, " pooled emb:", emb.shape, " pool:", pool)

        return emb

    estimates = []
    for _ in range(n_samples):
        if noise == "rademacher":
            r = _sample_rademacher_like(x_req)
        elif noise == "gaussian":
            r = torch.randn_like(x_req)
        else:
            raise ValueError(f"Unknown noise: {noise}")

        if normalize_r:
            # per-sample normalization (keeps batch structure)
            flat = r.view(r.shape[0], -1)
            norm = flat.norm(dim=1).clamp_min(eps).view(-1, *([1] * (r.dim() - 1)))
            r = r / norm

        # JVP: returns (f(x), J r)
        _, Jr = jvp(f, (x_req,), (r,), create_graph=False, strict=False)  # Jr: (B,D)

        # ||J r||^2 per sample
        e = (Jr ** 2).sum(dim=-1)  # (B,)
        estimates.append(e)

    trace_est = torch.stack(estimates, dim=0).mean(dim=0)  # (B,)
    return trace_est

# -----------------------------
# Utility: random probe vectors
# -----------------------------
def _rademacher(shape, device, dtype):
    return torch.empty(shape, device=device, dtype=dtype).bernoulli_(0.5).mul_(2).sub_(1)



# ------------------------------------------------------------
# Core: build matvec for A = J J^T in embedding space (D x D)
# using VJP (J^T u) + JVP (J v)
# ------------------------------------------------------------
def _make_A_matvec_JJt(
    encoder_fn: Callable[[torch.Tensor], torch.Tensor],
    x_single: torch.Tensor,                   # (C,H,W) or (C,T,H,W), requires_grad will be enabled inside
    pool: str = "mean",
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], int]:
    """
    Returns:
      matvec_A(u): computes (J J^T) u in R^D
      D: embedding dimension
    """

    # enable grad on x
    x_req = x_single.detach().requires_grad_(True)

    def f(inp: torch.Tensor) -> torch.Tensor:
        out = encoder_fn(inp.unsqueeze(0))      # -> (1,N,D) or (1,D)
        out = out.squeeze(0)                   # -> (N,D) or (D,)
        emb = _pool_tokens_if_needed(out, pool)  # (D,)
        return emb

    # get D
    with torch.no_grad():
        D = f(x_req).numel()

    def matvec_A(u: torch.Tensor) -> torch.Tensor:
        """
        u: (D,)
        returns: (D,) = (J J^T) u
        """
        # ---- VJP: v = J^T u  (same shape as x)
        emb = f(x_req)  # (D,)
        # scalar = <emb, u>
        s = torch.dot(emb, u)
        v = torch.autograd.grad(s, x_req, retain_graph=True, create_graph=False)[0]

        # ---- JVP: J v
        # jvp returns (f(x), Jv)
        _, Jv = jvp(f, (x_req,), (v,), create_graph=False, strict=False)  # (D,)
        return Jv

    return matvec_A, D


# -----------------------------
# Lanczos tridiagonalization
# -----------------------------
def _lanczos_tridiag(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    q1: torch.Tensor,     # (D,) normalized
    n_steps: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Returns T (n_steps x n_steps) tridiagonal matrix from Lanczos.
    """
    D = q1.numel()
    alphas = []
    betas = []

    q_prev = torch.zeros_like(q1)
    q = q1

    beta = torch.tensor(0.0, device=q.device, dtype=q.dtype)

    for j in range(n_steps):
        z = matvec(q)  # (D,)
        alpha = torch.dot(q, z)
        z = z - alpha * q - beta * q_prev

        beta = torch.linalg.norm(z).clamp_min(eps)
        alphas.append(alpha)

        if j < n_steps - 1:
            betas.append(beta)
            q_prev = q
            q = z / beta

    # build T
    T = torch.zeros((n_steps, n_steps), device=q1.device, dtype=q1.dtype)
    for i in range(n_steps):
        T[i, i] = alphas[i]
        if i < n_steps - 1:
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]

    return T


# ------------------------------------------------------------
# SLQ estimator for Tr(log(A + eps I)), A = JJ^T
# and return JEPA-SCORE = 0.5 * Tr(log(A + eps I))
# with torch.enable_grad():
#     with torch.cuda.amp.autocast(enabled=False):
#         score = jepa_score_slq(
#             encoder_fn=vjepa,
#             x=vid.float(),      # (B,C,T,H,W)
#             n_probe=8,
#             n_lanczos=20,
#             noise="rademacher",
#             pool="mean",
#             log_eps=1e-6,
#         )
# print("JEPA-SCORE (SLQ) =", score)
# ------------------------------------------------------------
def jepa_score_slq(
    encoder_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,  # (B,C,H,W) or (B,C,T,H,W)
    n_probe: int = 8,
    n_lanczos: int = 20,
    noise: Literal["rademacher", "gaussian"] = "rademacher",
    pool: str = "mean",
    log_eps: float = 1e-6,
) -> torch.Tensor:
    """
    Approximates JEPA-SCORE (Eq. 5): sum log sigma_k(J)
    using SLQ on A = J J^T (embedding space).

    Returns:
      score: (B,)  (approx) 0.5 * Tr(log(JJ^T + log_eps I))
    """
    if x.dim() not in (4, 5):
        raise ValueError(f"x must be (B,C,H,W) or (B,C,T,H,W). Got {x.shape}")

    B = x.shape[0]
    device = x.device
    dtype = x.dtype

    scores = []

    for b in range(B):
        x_single = x[b]  # (C,...) single sample

        matvec_A, D = _make_A_matvec_JJt(encoder_fn, x_single, pool=pool)

        # SLQ estimate of Tr(log(A + eps I)) = E_g [ g^T log(A+epsI) g ]
        # with probes g ~ N(0,I) or Rademacher
        ests = []
        for _ in range(n_probe):
            if noise == "rademacher":
                g = _rademacher((D,), device=device, dtype=dtype)
            elif noise == "gaussian":
                g = torch.randn((D,), device=device, dtype=dtype)
            else:
                raise ValueError(f"Unknown noise: {noise}")

            g_norm = torch.linalg.norm(g).clamp_min(1e-12)
            q1 = g / g_norm  # normalized starting vector

            # Lanczos on A
            T = _lanczos_tridiag(matvec_A, q1, n_steps=n_lanczos)

            # Gauss quadrature: g^T f(A) g ≈ ||g||^2 * e1^T f(T) e1
            # where e1=[1,0,...]
            evals, evecs = torch.linalg.eigh(T)  # T is symmetric tridiagonal
            # weights = (first component of eigenvectors)^2
            w = (evecs[0, :] ** 2)

            # f(λ)=log(λ+log_eps); add jitter for numerical stability
            f_evals = torch.log(evals.clamp_min(0.0) + log_eps)

            quad = (g_norm ** 2) * torch.sum(w * f_evals)  # scalar
            ests.append(quad)

        tr_log = torch.stack(ests).mean()  # scalar ≈ Tr(log(A+epsI))

        # JEPA-SCORE = 0.5 * Tr(log(JJ^T + eps I))  (since log σ = 0.5 log σ^2)
        scores.append(0.5 * tr_log)

    return torch.stack(scores, dim=0)  # (B,)
