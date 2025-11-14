import scanpy as sc
import numpy as np
import torch
from torch import nn, optim, autograd
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import shapely
from shapely import prepared
from shapely.geometry import Point, MultiPoint,  Polygon
from shapely.ops import nearest_points
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import pandas as pd
import os
from typing import Optional, Dict, List
# ─── 1. Data Preparation ──────────────────────────────────────────────────────
def prepare_coords(adata, lib, normalize_coords=True):
    """
    Extracts and (optionally) normalizes the spatial coordinates once.
    
    Returns:
      coords_t: FloatTensor of shape (N,2)
    """
    # 1) get low-res scale factor & raw pixel coords
    sf = adata.uns['spatial'][lib]['scalefactors']['tissue_lowres_scalef']
    coords = adata.obsm['spatial'] * sf  # (N,2)
    
    # 2) optional centering & scaling
    if normalize_coords:
        mean = coords.mean(0)
        std  = coords.std(0)
        coords = (coords - mean) / std
    
    # 3) to torch
    return torch.from_numpy(coords).float()


def prepare_weights(adata, gene, a_min = 1e-3):
    """
    Given an AnnData and a gene name, returns per-spot weights.
    
    Returns:
      weights_t: FloatTensor of shape (N,)
    """
    # 1) locate the gene
    try:
        gene_idx = adata.var_names.get_loc(gene)
    except KeyError:
        raise ValueError(f"Gene {gene!r} not found in adata.var_names")
    
    # 2) pull out its expression (handles both dense & sparse)
    f_vals = adata.X[:, gene_idx].toarray().ravel()  # (N,)
    
    # 3) build non-negative, normalized weights
    #w = np.exp(f_vals, dtype=float)
    w = f_vals.astype(float)  # ensure float type
    w = np.clip(w, a_min=a_min, a_max=None)
    
    print(max(w), min(w))
    w = w / w.sum()
    
    # 4) to torch
    return torch.from_numpy(w).float()


def boundary_distances_and_gradients_torch(
    coords,
    ratio: float = 0.3,
    eps: float = 0.01,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    visualize: bool = True):
    """s
    coords: (N,2) array or tensor of x,y positions
    ratio: concavity for shapely.concave_hull (0<ratio<=1)
    eps: outward buffer so no point lies exactly on boundary
    device: where to put the output tensors
    dtype: numeric type for the outputs
    
    Returns:
      distances:   (N,)   torch tensor of floats
      gradients:   (N,2) torch tensor of unit vectors
      bndry_pts:   (N,2) torch tensor of the closest boundary coords
    """
    # 1) Bring coords into NumPy
    if isinstance(coords, torch.Tensor):
        coords_np = coords.detach().cpu().numpy()
    else:
        coords_np = np.asarray(coords)
    
    # 2) Build hull (and optional buffer)
    mp = MultiPoint(coords_np)
    hull = shapely.concave_hull(mp, ratio=ratio)
    if eps > 0:
        hull = hull.buffer(eps, join_style=2)
    boundary = hull.exterior
    
    # 3) Allocate arrays
    N = coords_np.shape[0]
    dists = np.zeros(N, dtype=float)
    grads = np.zeros((N, 2), dtype=float)
    bpts  = np.zeros((N, 2), dtype=float)
    
    # 4) Compute for each point
    for i, (x, y) in enumerate(coords_np):
        pt = Point(x, y)
        _, nb = nearest_points(pt, boundary)
        bx, by = nb.x, nb.y
        
        dist = np.hypot(x - bx, y - by)
        dists[i] = dist
        bpts[i]  = [bx, by]
        
        if dist > 0:
            grads[i] = [(x - bx) / dist, (y - by) / dist]
        else:
            grads[i] = [0.0, 0.0]  # or np.nan if you prefer
    
    # 4a) Visualize the boundary if requested
    if visualize:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        # — spots + boundary
        ax[0].scatter(coords_np[:,0], coords_np[:,1], s=5)
        xb, yb = hull.exterior.xy
        ax[0].plot(xb, yb,'r-', linewidth=1.5)
        ax[0].set_aspect('equal', 'box')
        ax[0].set_title("Spots and boundary")
        # — distance heatmap
        sc = ax[1].scatter(coords_np[:,0], coords_np[:,1], c=dists)
        fig.colorbar(sc, ax=ax[1], label="Distance to boundary")
        ax[1].set_aspect('equal', 'box')
        ax[1].set_title("Distance to boundary")
        plt.tight_layout()
        plt.show()

    # 5) Convert back to torch
    device = device or torch.device('cpu')
    t_dists = torch.from_numpy(dists).to(device=device, dtype=dtype)
    t_grads = torch.from_numpy(grads).to(device=device, dtype=dtype)
    t_bpts  = torch.from_numpy(bpts).to(device=device, dtype=dtype)
    
    return t_dists, t_grads, t_bpts, hull




# ─── 2. Model Definition ──────────────────────────────────────────────────────
# ---------- Activations ----------
def make_activation(name: str = "tanh", beta: float = 5.0):
    name = name.lower()
    if name == "tanh":     return nn.Tanh()
    if name == "silu":     return nn.SiLU()
    if name == "gelu":     return nn.GELU()
    if name == "softplus": return nn.Softplus(beta=beta)  # C^∞, choose beta∈[1,5]
    raise ValueError(f"Unknown activation {name}")

# ---------- Optional Fourier features for detail without jagged second-derivs ----------
class FourierFeatures(nn.Module):
    """
    Fixed random Fourier features: x -> [sin(Bx), cos(Bx)], B ~ N(0, sigma^2).
    Keep sigma small for smoother fields; larger for finer detail.
    """
    def __init__(self, in_dim=2, num_feats=32, sigma=3.0, bias=True):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_dim, num_feats) * sigma, requires_grad=False)
        self.b = nn.Parameter(torch.zeros(num_feats), requires_grad=False) if bias else None
        self.out_dim = 2 * num_feats

    def forward(self, x):  # x: (B,2)
        z = x @ self.B  # (B, num_feats)
        if self.b is not None:
            z = z + self.b
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)

# ---------- Residual MLP block ----------
class ResBlock(nn.Module):
    def __init__(self, dim, act):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act  = act
    def forward(self, x):
        h = self.act(self.lin1(x))
        h = self.lin2(h)
        return self.act(x + h)  # preact-residual; smooth if act is smooth

# ---------- Potential network: f(x) ≈ log λ(x) ----------
class LogLambdaNet(nn.Module):
    def __init__(self,
                 hidden_dim=128,
                 depth=3,
                 activation="silu",
                 use_fourier=False,
                 fourier_feats=32,
                 fourier_sigma=3.0):
        super().__init__()
        act = make_activation(activation)

        in_dim = 2
        if use_fourier:
            self.fe = FourierFeatures(in_dim=2, num_feats=fourier_feats, sigma=fourier_sigma)
            in_dim = self.fe.out_dim
        else:
            self.fe = None

        self.inp  = nn.Linear(in_dim, hidden_dim)
        self.act  = act
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, act) for _ in range(max(0, depth-1))])
        self.out  = nn.Linear(hidden_dim, 1)

        # small final init to stabilize Laplacians early
        nn.init.uniform_(self.out.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        if self.fe is not None:
            x = self.fe(x)
        h = self.act(self.inp(x))
        for blk in self.blocks:
            h = blk(h)
        return self.out(h).squeeze(-1)  # (B,)

# ---------- Score network: s(x) ≈ ∇ log λ(x) ----------
class ScoreNet(nn.Module):
    def __init__(self,
                 hidden_dim=128,
                 depth=3,
                 activation="silu",
                 use_fourier=False,
                 fourier_feats=32,
                 fourier_sigma=3.0):
        super().__init__()
        act = make_activation(activation)

        in_dim = 2
        if use_fourier:
            self.fe = FourierFeatures(in_dim=2, num_feats=fourier_feats, sigma=fourier_sigma)
            in_dim = self.fe.out_dim
        else:
            self.fe = None

        self.inp  = nn.Linear(in_dim, hidden_dim)
        self.act  = act
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, act) for _ in range(max(0, depth-1))])
        self.out  = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        if self.fe is not None:
            x = self.fe(x)
        h = self.act(self.inp(x))
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)  # (B,2)


# ─── 3. Implicit SM Loss ──────────────────────────────────────────────────────
# def riemann_ism_loss(model, coords, weights):
#     """
#     Hyvärinen’s implicit score matching via weighted Riemann sum.
#     coords: (B,2) tensor with requires_grad=True
#     weights: (B,) tensor
#     """
#     coords = coords.requires_grad_(True)
#     s = model(coords)                # (B,2)

#     # compute divergence = ∂s1/∂x + ∂s2/∂y
#     div_terms = []
#     for dim in range(2):
#         g = autograd.grad(
#             outputs=s[:, dim].sum(),
#             inputs=coords,
#             create_graph=True
#         )[0][:, dim]
#         div_terms.append(g)
#     divergence = div_terms[0] + div_terms[1]  # (B,)

#     norm2 = (s**2).sum(dim=1)        # (B,)
#     loss  = ((norm2 + 2 * divergence) * weights).sum()
#     return loss
class KNNGate:
    """
    Precompute kNN of spots once; reuse to build gates per gene.
    - spot_xy: (N,2) array of spot coordinates (already normalized if you like)
    - k: number of neighbors to use for the gate
    - include_self: if True and queries are the same spots, the 1st neighbor is self;
      we drop it automatically so you still get k *other* neighbors.
    """
    def __init__(self, spot_xy, k=8, include_self=False,
                 metric="euclidean", algorithm="auto"):
        self.xy = np.asarray(spot_xy, dtype=np.float32)
        if self.xy.ndim != 2 or self.xy.shape[1] != 2:
            raise ValueError("spot_xy must be an (N,2) array")
        self.N = self.xy.shape[0]
        self.k = int(k)
        self.include_self = bool(include_self)

        # Build NN index on spots
        n_neighbors = self.k + (1 if self.include_self else 0)
        self.nn = NearestNeighbors(n_neighbors=n_neighbors,
                                   algorithm=algorithm, metric=metric)
        self.nn.fit(self.xy)

        # Precompute neighbors for the spots themselves
        dists, idx = self.nn.kneighbors(self.xy, return_distance=True)
        if self.include_self:
            # Drop the self index per row, then take the first k remaining
            neigh = []
            for i in range(self.N):
                row = idx[i]
                row_wo_self = row[row != i]
                neigh.append(row_wo_self[:self.k])
            self.neigh_idx_spots = np.stack(neigh, axis=0)  # (N,k)
        else:
            self.neigh_idx_spots = idx[:, :self.k]          # (N,k)

    # ---------- Gates on SPOTS (reuse for every gene) ----------

    def gate_hard_spots(self, counts, tau=0.0):
        """
        Proportion of zero (<= tau) counts among k neighbors, per spot.
        counts: (N,) array for a single gene (already preprocessed if desired).
        Returns r: (N,) in [0,1]
        """
        c = np.asarray(counts, dtype=np.float32)
        if c.shape != (self.N,):
            raise ValueError(f"counts must be shape ({self.N},)")
        neigh_c = c[self.neigh_idx_spots]               # (N,k)
        r = (neigh_c <= float(tau)).mean(axis=1)        # (N,)
        return torch.tensor(r.astype(np.float32))

    def gate_soft_spots(self, counts, kappa=1.0):
        """
        Soft gate using 2*sigmoid(-c/kappa), averaged over neighbors, per spot.
        counts: (N,) array for a single gene.
        kappa: softness (scale of counts); smaller = sharper step near 0.
        Returns r: (N,) in (0,1]
        """
        c = np.asarray(counts, dtype=np.float32)
        if c.shape != (self.N,):
            raise ValueError(f"counts must be shape ({self.N},)")
        neigh_c = c[self.neigh_idx_spots]               # (N,k)
        r = (2.0 / (1.0 + np.exp(neigh_c / float(kappa)))).mean(axis=1)
        return torch.tensor(r.astype(np.float32))

    # ---------- Gates on ARBITRARY QUERY POINTS (optional) ----------

    def gate_hard_queries(self, query_xy, counts, tau=0.0):
        """
        Same gate but for arbitrary query points Z (e.g., uniform probes).
        Uses the spot kNN index; neighbors are spots (and never the query).
        """
        c = np.asarray(counts, dtype=np.float32)
        if c.shape != (self.N,):
            raise ValueError(f"counts must be shape ({self.N},)")
        Z = np.asarray(query_xy, dtype=np.float32)
        _, idx = self.nn.kneighbors(Z, return_distance=True)  # (M, k or k+1)
        if self.include_self:
            # For queries not equal to any spot, no true "self"; still safe to take first k
            idx = idx[:, :self.k]
        neigh_c = c[idx]                                     # (M,k)
        r = (neigh_c <= float(tau)).mean(axis=1)
        return torch.tensor(r.astype(np.float32))

    def gate_soft_queries(self, query_xy, counts, kappa=1.0):
        c = np.asarray(counts, dtype=np.float32)
        if c.shape != (self.N,):
            raise ValueError(f"counts must be shape ({self.N},)")
        Z = np.asarray(query_xy, dtype=np.float32)
        _, idx = self.nn.kneighbors(Z, return_distance=True)
        if self.include_self:
            idx = idx[:, :self.k]
        neigh_c = c[idx]
        r = (2.0 / (1.0 + np.exp(neigh_c / float(kappa)))).mean(axis=1)
        return torch.tensor(r.astype(np.float32))

def _divergence(s, coords):
    # s: (B,2), coords: (B,2) with requires_grad=True
    div_terms = []
    for k in range(2):
        gk = autograd.grad(
            outputs=s[:, k].sum(),
            inputs=coords,
            create_graph=True
        )[0][:, k]
        div_terms.append(gk)
    return div_terms[0] + div_terms[1]



def riemann_truncsm_loss(
    model: torch.nn.Module,
    coords: torch.Tensor,          # (B,2)
    weights: torch.Tensor,         # (B,)
    distances: torch.Tensor,       # (B,) = g0(x)
    grad_g0: torch.Tensor,         # (B,2) = ∇g0(x)
    # ----- gated regularizer (applies in both modes) -----
    gate_alpha: float = 0.0,                       # strength
    gate_r_on_coords: Optional[torch.Tensor] = None,  # (B,) gate r(x) for coords
    reg_coords: Optional[torch.Tensor] = None,        # (M,2) optional extra query points
    reg_gate_r: Optional[torch.Tensor] = None,        # (M,) gate on reg_coords
    reg_weights: Optional[torch.Tensor] = None,       # (M,) weights for reg term (if provided)
    # For potential mode only: push f(x) negative via hinge on positive part
    gate_margin: float = 0.0,                      # hinge margin m; penalize (f - m)_+^2
    # ----- curl regularization (score mode only) -----
    curl_eta: float = 0.0,
):
    """
    Unified truncated score-matching + optional regularizers.

    Auto-detect mode by model(coords).shape:
      - Potential mode (log-intensity): (B,) or (B,1)
          s = ∇f,  div s = Δ f
          Gate regularizer penalizes positive f in gated regions:
              L_gate ~ E[ r(x) * (ReLU(f(x) - m))^2 ]
      - Score mode: (B,2)
          s = model(coords),  div s from autograd
          Gate regularizer penalizes score magnitude in gated regions:
              L_gate ~ E[ r(x) * ||s(x)||^2 ]
          Optional curl penalty:
              L_curl ~ E[ (∂s_y/∂x - ∂s_x/∂y)^2 ]

    The core TSM objective is:
        J ≈ Σ_i w_i [ g0_i ||s||^2 + 2 g0_i div s + 2 ⟨∇g0_i, s⟩ ].
    """
    # --- ensure device / dtype alignment ---
    dev   = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    coords    = coords.to(dev, dtype=dtype).requires_grad_(True)
    weights   = weights.to(dev, dtype=dtype)
    distances = distances.to(dev, dtype=dtype)
    grad_g0   = grad_g0.to(dev, dtype=dtype)

    if gate_r_on_coords is not None:
        gate_r_on_coords = gate_r_on_coords.to(dev, dtype=dtype)
    if reg_coords is not None:
        reg_coords = reg_coords.to(dev, dtype=dtype)
    if reg_gate_r is not None:
        reg_gate_r = reg_gate_r.to(dev, dtype=dtype)
    if reg_weights is not None:
        reg_weights = reg_weights.to(dev, dtype=dtype)

    # --- forward & mode detection ---
    out = model(coords)
    if out.ndim == 1:
        out = out.unsqueeze(-1)
    if out.ndim != 2:
        raise ValueError(f"Model output must be (B,2) or (B,1)/(B,); got {tuple(out.shape)}")

    potential_mode = (out.shape[1] == 1)

    # --- build s and divergence ---
    if potential_mode:
        f = out[:, 0]  # (B,)
        s = autograd.grad(f.sum(), coords, create_graph=True)[0]     # (B,2)
        divergence = _divergence(s, coords)                          # (B,)
    else:
        s = out                                                     # (B,2)
        divergence = _divergence(s, coords)                          # (B,)

    # --- core TSM terms ---
    norm2 = (s ** 2).sum(dim=1)                 # ||s||^2
    dot   = (grad_g0 * s).sum(dim=1)            # <∇g0, s>
    loss_terms = distances * norm2 + 2.0 * distances * divergence + 2.0 * dot
    loss = (loss_terms * weights).sum()

    # ---------------- Gate regularizers ----------------
    if gate_alpha > 0.0:
        if potential_mode:
            # On the same coords: penalize positive part of f (toward negative)
            if gate_r_on_coords is not None:
                hinge = f - gate_margin           # (B,)
                L_gate_coords = gate_alpha * (gate_r_on_coords * (hinge ** 2)).mean()
                loss = loss + L_gate_coords

            # On separate query points (optional)
            if (reg_coords is not None) and (reg_gate_r is not None):
                reg_coords_req = reg_coords.detach().requires_grad_(False)
                f_reg = model(reg_coords_req)
                if f_reg.ndim == 2 and f_reg.shape[1] == 1:
                    f_reg = f_reg[:, 0]
                elif f_reg.ndim == 1:
                    pass
                else:
                    # If user mistakenly passes a score-model here, coerce to potential penalty on its implicit f is undefined.
                    # Fall back: build f via a tiny backward? Safer to raise for clarity.
                    raise ValueError("Gate reg for potential mode expects a potential model; got 2D score output on reg_coords.")
                hinge_reg = torch.relu(f_reg - gate_margin)   # (M,)
                if reg_weights is not None:
                    L_gate_reg = gate_alpha * (reg_gate_r * (hinge_reg ** 2) * reg_weights).sum() / (reg_weights.sum() + 1e-12)
                else:
                    L_gate_reg = gate_alpha * (reg_gate_r * (hinge_reg ** 2)).mean()
                loss = loss + L_gate_reg

        else:
            # Score mode: penalize magnitude ||s||^2 in gated regions
            if gate_r_on_coords is not None:
                L_gate_coords = gate_alpha * (gate_r_on_coords * norm2).mean()
                loss = loss + L_gate_coords

            if (reg_coords is not None) and (reg_gate_r is not None):
                s_reg = model(reg_coords)                       # (M,2)
                norm2_reg = (s_reg ** 2).sum(dim=1)             # (M,)
                if reg_weights is not None:
                    L_gate_reg = gate_alpha * (reg_gate_r * norm2_reg * reg_weights).sum() / (reg_weights.sum() + 1e-12)
                else:
                    L_gate_reg = gate_alpha * (reg_gate_r * norm2_reg).mean()
                loss = loss + L_gate_reg

    # ---------------- Curl regularizer (score mode only) ----------------
    if (not potential_mode) and (curl_eta > 0.0):
        pts = (reg_coords if reg_coords is not None else coords).detach().requires_grad_(True)
        s_pts = model(pts)                                  # (M,2) or (B,2)
        sx, sy = s_pts[:, 0], s_pts[:, 1]
        grad_sx = autograd.grad(sx.sum(), pts, create_graph=True)[0]   # ∇s_x
        grad_sy = autograd.grad(sy.sum(), pts, create_graph=True)[0]   # ∇s_y
        curl = grad_sy[:, 0] - grad_sx[:, 1]                               # scalar curl in 2D
        L_curl = curl_eta * (curl ** 2).mean()
        loss = loss + L_curl

    return loss


# ─── 4. Training Loop ─────────────────────────────────────────────────────────
def train(
    model: torch.nn.Module,
    coords: torch.Tensor,
    weights: torch.Tensor,
    distances: torch.Tensor,
    grad_g0: torch.Tensor,
    lr: float = 1e-3,
    weights_decay: float = 0.0,
    epochs: int = 500,
    step_size: int = 200,
    gamma: float = 0.1,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    # ----- gate regularizer knobs -----
    gate_alpha: float = 0.0,                       # α (0 disables)
    gate_r: Optional[torch.Tensor] = None,         # (N,) precomputed r(x_i) for spots
    # optional: separate query points and their gates
    reg_coords: Optional[torch.Tensor] = None,     # (M,2)
    reg_gate_r: Optional[torch.Tensor] = None,     # (M,)
    reg_weights: Optional[torch.Tensor] = None,    # (M,)
    # margin for potential mode gate (hinge on positive part of f - margin)
    gate_margin: float = 0.0,
    # ----- curl regularization (score mode only) -----
    curl_eta: float = 0.0,
    # ----- extras -----
    grad_clip: Optional[float] = None,             # e.g., 1.0 to clip global norm
    log_every: int = 50,
) -> List[float]:
    """
    Train with truncated score matching + optional gate/curl regularizers.
    Compatible with *both* potential-mode (scalar f) and score-mode (2D s) models,
    as auto-detected inside `riemann_truncsm_loss`.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device)

    # Move base tensors (reg_* tensors are reused each step, so move once here)
    coords    = coords.to(device)
    weights   = weights.to(device)
    distances = distances.to(device)
    grad_g0   = grad_g0.to(device)
    if gate_r is not None:
        gate_r = gate_r.to(device)

    if reg_coords is not None:
        reg_coords = reg_coords.to(device)
    if reg_gate_r is not None:
        reg_gate_r = reg_gate_r.to(device)
    if reg_weights is not None:
        reg_weights = reg_weights.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weights_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Build loader
    if batch_size:
        if gate_r is not None:
            dataset = TensorDataset(coords, weights, distances, grad_g0, gate_r)
        else:
            dataset = TensorDataset(coords, weights, distances, grad_g0)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        if gate_r is not None:
            loader = [(coords, weights, distances, grad_g0, gate_r)]
        else:
            loader = [(coords, weights, distances, grad_g0)]

    loss_history: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            optimizer.zero_grad(set_to_none=True)

            if gate_r is not None:
                cb, wb, db, gg, rb = batch
            else:
                cb, wb, db, gg = batch
                rb = None

            # Unified loss call (auto-detects potential vs score model)
            loss = riemann_truncsm_loss(
                model, cb, wb, db, gg,
                gate_alpha=gate_alpha,
                gate_r_on_coords=rb,       # per-spot gate on the same coords
                reg_coords=reg_coords,     # optional separate query points
                reg_gate_r=reg_gate_r,
                reg_weights=reg_weights,
                gate_margin=gate_margin,   # only used in potential mode
                curl_eta=curl_eta,         # only used in score mode
            )
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            epoch_loss += float(loss.detach().cpu())

        loss_history.append(epoch_loss)

        if epoch == 1 or (log_every and epoch % log_every == 0):
            print(f"[Epoch {epoch}/{epochs}] loss = {epoch_loss:.4f}")

        scheduler.step()
# ─── 5. Get Model Output ───────────────────────────────────────────────────
@torch.inference_mode(False)  # we need grads for potential mode
def get_model_outputs(
    model: torch.nn.Module,
    coords: torch.Tensor,               # (N,2)
    batch_size: int = 4096,
    device: Optional[torch.device] = None,
    to_cpu: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Auto-detects model type by output shape and returns:
      - Score model (B,2): {"score": s} with s shape (N,2)
      - Potential model (B,) or (B,1):
          {
            "log_potential": f,          # (N,)   f ≈ log λ + C (no gauge fix)
            "grad_log_potential": df,    # (N,2)  ∇f
            "potential": lam,            # (N,)   exp(f)
            "grad_potential": dlam       # (N,2)  exp(f) * ∇f
          }

    Notes:
      • No gauge fix is applied. If you want ∫λ̂=N_obs, compute C separately and add to f.
      • Returns CPU tensors by default (to_cpu=True).
    """
    device = device or next(model.parameters()).device
    N = coords.shape[0]

    # Probe to detect mode
    with torch.no_grad():
        probe = model(coords[:2].to(device))
    # Normalize shape
    if probe.ndim == 1:
        probe_shape = (probe.shape[0], 1)
    else:
        probe_shape = tuple(probe.shape)

    is_potential = (probe_shape[1] == 1)
    out: Dict[str, torch.Tensor] = {}

    if not is_potential:
        # -------- Score model: just return s(x) --------
        chunks = []
        model.eval()
        with torch.no_grad():
            for i in range(0, N, batch_size):
                xb = coords[i:i+batch_size].to(device)
                sb = model(xb)                 # (B,2)
                chunks.append(sb.detach().cpu() if to_cpu else sb.detach())
        s = torch.cat(chunks, dim=0)
        out["score"] = s
        return out

    # -------- Potential model: need grads --------
    f_list, df_list, lam_list, dlam_list = [], [], [], []
    model.eval()
    for i in range(0, N, batch_size):
        xb = coords[i:i+batch_size].to(device).detach().requires_grad_(True)
        fb = model(xb).squeeze(-1)                       # (B,)
        dfb = autograd.grad(fb.sum(), xb, create_graph=False)[0]  # (B,2)
        lamb = torch.exp(fb)                              # (B,)
        dlam = lamb.unsqueeze(-1) * dfb                   # (B,2)

        if to_cpu:
            f_list.append(fb.detach().cpu())
            df_list.append(dfb.detach().cpu())
            lam_list.append(lamb.detach().cpu())
            dlam_list.append(dlam.detach().cpu())
        else:
            f_list.append(fb.detach())
            df_list.append(dfb.detach())
            lam_list.append(lamb.detach())
            dlam_list.append(dlam.detach())

    out["log_potential"]     = torch.cat(f_list, dim=0)     # (N,)
    out["grad_log_potential"] = torch.cat(df_list, dim=0)   # (N,2)
    out["potential"]         = torch.cat(lam_list, dim=0)   # (N,)
    out["grad_potential"]    = torch.cat(dlam_list, dim=0)  # (N,2)
    return out

# ─── 6. Langevin Dynamics for Sampling ───────────────────────────────────────────────────



# def _project_outside_to_inside(poly, x_np, push_in=1e-3):
#     """
#     For any row in x_np outside `poly`, project to the nearest boundary point,
#     then move a small distance `push_in` along the inward normal (toward interior).
#     """
#     prep = prepared.prep(poly)
#     out_idx = [i for i, (xx, yy) in enumerate(x_np) if not prep.contains(Point(xx, yy))]
#     if not out_idx:
#         return x_np

#     for i in out_idx:
#         px, py = x_np[i]
#         pt = Point(px, py)
#         _, nb = nearest_points(pt, poly.exterior)
#         bx, by = nb.x, nb.y
#         v = np.array([px - bx, py - by], dtype=float)
#         nrm = np.linalg.norm(v) + 1e-12
#         n_out = v / nrm              # outward normal (boundary -> outside point)
#         n_in  = -n_out               # inward normal
#         x_np[i] = np.array([bx, by]) + n_in * push_in
#     return x_np


# @torch.no_grad()
# def sample_ld_in_polygon(
#     model,
#     poly,                         # Shapely Polygon
#     M: int = 5000,
#     T: int = 1000,
#     eps: float = 0.01,            # step size (smaller = more refined)
#     sigma_noise: float = 1.0,
#     alpha: float = 0.999,         # noise annealing rate
#     device: str = "cpu",
#     dtype: torch.dtype = torch.float32,
#     push_in: float = 1e-3,        # how far to push inside after projection
#     init: str = "uniform"         # "uniform" or "reuse" (if you pass your own x0)
# ):
#     """
#     Projected-Langevin sampling constrained to a polygonal domain.
#     Keeps all samples inside `poly` by projection after each step.
#     """
#     # 0) init inside the polygon
#     if init == "uniform":
#         x = _uniform_in_polygon(poly, M, device=device, dtype=dtype)
#     else:
#         raise ValueError("Unknown init mode. Use 'uniform' or extend as needed.")

#     # 1) dynamics
#     for t in range(T):
#         sigma_t = sigma_noise * (alpha ** t)

#         s = model(x)
#         noise = torch.randn_like(x) * sigma_t

#         x_prop = x + 0.5 * (eps ** 2) * s + eps * noise

#         # 2) project back any outside points
#         x_np = x_prop.detach().cpu().numpy().astype(np.float32)
#         x_np = _project_outside_to_inside(poly, x_np, push_in=push_in)

#         x = torch.from_numpy(x_np).to(device=device, dtype=dtype)

#     return x  # (M,2) inside the boundary


def rasterize_polygon_sdf(
    poly: Polygon,
    H: int = 1024,
    W: int = 1024,
    margin: float = 0.02,
    dtype=np.float32,
    boundary_as_inside: bool = False,
):
    """
    Build a signed-distance field (SDF) for `poly` on an HxW grid over a padded bbox.
    The SDF is in *world units* (negative inside, ~0 on boundary, positive outside)
    because the EDT is informed of true pixel sizes via `sampling=(sy, sx)`.

    Args
    ----
    poly : shapely.geometry.Polygon
        Target polygon (can be non-convex; holes allowed).
    H, W : int
        Grid resolution: H rows (y), W cols (x).
    margin : float
        Extra padding added to the bbox as a fraction of the bbox size
        (applied before computing the grid).
    dtype : np.dtype
        Floating dtype for outputs.
    boundary_as_inside : bool
        If True, treat boundary points as inside (uses `covers` test if available);
        otherwise uses `contains` (boundary excluded).

    Returns
    -------
    sdf : (H, W) ndarray of dtype
        Signed distance in world units (neg inside, pos outside).
    xs  : (W,) ndarray of dtype
        World x-coordinates of grid columns.
    ys  : (H,) ndarray of dtype
        World y-coordinates of grid rows.
    """
    assert isinstance(poly, Polygon), "poly must be a Shapely Polygon"

    # --- padded bbox in world coords ---
    minx, miny, maxx, maxy = poly.bounds
    dx, dy = maxx - minx, maxy - miny
    minx -= margin * dx; maxx += margin * dx
    miny -= margin * dy; maxy += margin * dy

    # grid coords (world)
    xs = np.linspace(minx, maxx, W, dtype=dtype)
    ys = np.linspace(miny, maxy, H, dtype=dtype)
    xx, yy = np.meshgrid(xs, ys)  # (H, W)

    # pixel size in world units (note: rows<->y uses sy; cols<->x uses sx)
    sx = (maxx - minx) / max(1, W - 1)
    sy = (maxy - miny) / max(1, H - 1)

    # --- inside mask ---
    # contains() is False on boundary; covers() is True on boundary.
    # prepared geometry speeds up many point-in-polygon tests.
    prep = prepared.prep(poly)
    if boundary_as_inside and hasattr(prep, "covers"):
        inside_iter = (prep.covers(Point(xy)) for xy in zip(xx.ravel(), yy.ravel()))
    else:
        inside_iter = (prep.contains(Point(xy)) for xy in zip(xx.ravel(), yy.ravel()))
    inside = np.fromiter(inside_iter, count=H * W, dtype=np.uint8).reshape(H, W)

    # --- Euclidean distance transforms in *world units* directly ---
    # distance_transform_edt returns distance to nearest zero of the input.
    # We pass sampling=(sy, sx) to reflect anisotropic pixel sizes (rows=y, cols=x).
    dist_out = distance_transform_edt(1 - inside, sampling=(sy, sx))  # to nearest inside
    dist_in  = distance_transform_edt(inside,       sampling=(sy, sx))  # to nearest outside

    # Signed distance (positive outside, negative inside), already in world units.
    sdf = (dist_out - dist_in).astype(dtype)

    return sdf, xs, ys


# ==========================================================================
# Step 2. A GPU projector that queries phi and grad phi by bilinear sampling
# ==========================================================================
class SDFProjector(torch.nn.Module):
    """
    Hold an SDF grid on device and provide:
      - phi(x): approximate signed distance at arbitrary world coords
      - grad(x): approximate gradient via finite differences
      - project(x): one (or a few) Newton-like steps to boundary
    """
    def __init__(self, sdf: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                 device: str = "cuda", dtype: torch.dtype = torch.float32):
        super().__init__()
        assert sdf.ndim == 2, "sdf must be (H, W)"
        self.H, self.W = sdf.shape
        self.register_buffer("sdf_grid", torch.from_numpy(sdf)[None, None].to(device=device, dtype=dtype))
        # Store bounds and scales for world <-> grid_sample mapping
        self.xmin = float(xs[0]); self.xmax = float(xs[-1])
        self.ymin = float(ys[0]); self.ymax = float(ys[-1])
        self.device = device
        self.dtype = dtype

        # Precompute world pixel sizes (for gradient scaling)
        self.dx = (self.xmax - self.xmin) / max(1, self.W - 1)
        self.dy = (self.ymax - self.ymin) / max(1, self.H - 1)

        # Small normalized offsets for central differences in grid space
        self.epsx = 2.0 / max(1, self.W - 1)
        self.epsy = 2.0 / max(1, self.H - 1)

    def _to_gridcoords(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map world coords x: (N,2) to grid_sample coords in [-1,1]: (1,N,1,2)
        """
        gx = 2.0 * (x[:, 0] - self.xmin) / max(1e-12, (self.xmax - self.xmin)) - 1.0
        gy = 2.0 * (x[:, 1] - self.ymin) / max(1e-12, (self.ymax - self.ymin)) - 1.0
        return torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)

    @torch.no_grad()
    def phi_and_grad(self, x: torch.Tensor):
        """
        x: (N,2) world coordinates on self.device
        returns:
          phi:  (N,)   approximate signed distance
          grad: (N,2)  approximate gradient in world units
        """
        # Bilinear interpolation for phi
        grid = self._to_gridcoords(x)
        phi = F.grid_sample(self.sdf_grid, grid, mode="bilinear",
                            padding_mode="border", align_corners=True).view(-1)

        # Central differences in normalized grid -> scale to world
        offs_x = torch.tensor([[[[self.epsx, 0.0]]]], device=x.device, dtype=x.dtype)
        offs_y = torch.tensor([[[[0.0, self.epsy]]]], device=x.device, dtype=x.dtype)

        gx_p = F.grid_sample(self.sdf_grid, grid + offs_x, align_corners=True).view(-1)
        gx_m = F.grid_sample(self.sdf_grid, grid - offs_x, align_corners=True).view(-1)
        gy_p = F.grid_sample(self.sdf_grid, grid + offs_y, align_corners=True).view(-1)
        gy_m = F.grid_sample(self.sdf_grid, grid - offs_y, align_corners=True).view(-1)

        # Convert to world derivatives
        dphidx = (gx_p - gx_m) / (2.0 * self.dx)
        dphidy = (gy_p - gy_m) / (2.0 * self.dy)
        grad = torch.stack([dphidx, dphidy], dim=-1)
        return phi, grad

    @torch.no_grad()
    def project(self, x: torch.Tensor, iters: int = 1, push_in: float = 1e-3):
        """
        Project any point that is outside (phi>0) onto the boundary (phi≈0),
        then push a tiny step along -grad to ensure it's inside.
        Only modifies violators; returns updated x.
        """
        for _ in range(iters):
            phi, grad = self.phi_and_grad(x)
            mask = phi > 0
            if not mask.any():
                break
            g2 = (grad[mask] * grad[mask]).sum(-1).clamp_min(1e-12)
            # Newton-like step to boundary
            x[mask] = x[mask] - (phi[mask] / g2).unsqueeze(-1) * grad[mask]
            # small push just inside along -grad direction
            step = (push_in / g2.sqrt()).unsqueeze(-1) * grad[mask]
            x[mask] = x[mask] - step
        return x


# ============================================================
# Step 3. Langevin-with-projection, fully on GPU in the hot path
# ============================================================
# ---------- 1) One-time: build a reusable projector ----------
def build_sdf_projector(poly,
                        H: int = 1024,
                        W: int = 1024,
                        margin: float = 0.02,
                        device: str = "cuda",
                        dtype: torch.dtype = torch.float32,
                        boundary_as_inside: bool = False) -> SDFProjector:
    """
    Rasterize the polygon SDF once (CPU) and return a GPU SDFProjector you can reuse.

    Args:
      poly: shapely Polygon
      H, W: grid resolution (rows, cols)
      margin: bbox padding (fraction of bbox size)
      device/dtype: where to keep the SDF grid for fast GPU queries
      boundary_as_inside: if True, treat boundary as inside for the mask

    Returns:
      SDFProjector, ready for phi/grad/project queries on `device`.
    """
    sdf, xs, ys = rasterize_polygon_sdf(
        poly, H=H, W=W, margin=margin, dtype=np.float32,
        boundary_as_inside=boundary_as_inside
    )
    return SDFProjector(sdf, xs, ys, device=device, dtype=dtype)


# ---------- 2) Sampling: reuse the projector (no rasterization inside) ----------
@torch.no_grad()
def _uniform_in_polygon(poly, M, device="cpu", dtype=torch.float32, batch=4096, rng=None):
    """Uniformly initialize M points inside a Shapely polygon via rejection sampling."""
    rng = np.random.default_rng() if rng is None else rng
    minx, miny, maxx, maxy = poly.bounds
    prep = prepared.prep(poly)
    pts = []

    while len(pts) < M:
        k = max(batch, M - len(pts))
        xs = rng.uniform(minx, maxx, size=k)
        ys = rng.uniform(miny, maxy, size=k)
        cand = np.stack([xs, ys], axis=1)
        mask = [prep.contains(Point(x, y)) for x, y in cand]
        if any(mask):
            pts.extend(cand[np.array(mask)].tolist())

    arr = np.asarray(pts[:M], dtype=np.float32)
    return torch.from_numpy(arr).to(device=device, dtype=dtype)

@torch.no_grad()
def sample_ld_in_polygon_sdf(model,
                             projector: SDFProjector,
                             M: int = 5000,
                             T: int = 1000,
                             eps: float = 0.01,
                             sigma_noise: float = 1.0,
                             alpha: float = 0.999,
                             project_every: int = 5,
                             project_iters: int = 1,
                             push_in: float = 1e-3,
                             init_x: torch.Tensor = None,
                             init_sampler=None,
                             poly=None,
                             dtype: torch.dtype = None):
    """
    Projected-Langevin sampling constrained by a prebuilt SDF projector.

    Args:
      model:         callable mapping (N,2) -> (N,2) (score/drift)
      projector:     SDFProjector built once via build_sdf_projector(...)
      M, T:          number of samples and LD steps
      eps:           LD step size
      sigma_noise:   initial noise scale
      alpha:         per-step noise decay (annealing)
      project_every: project only every K steps (and at the last step)
      project_iters: Newton-like correction steps per projection (1–2 is enough)
      push_in:       small inward push after projection to ensure interior
      init_x:        optional (M,2) tensor to warm start (must be on projector.device)
      init_sampler:  optional callable(M) -> (M,2) array/tensor to initialize
      poly:          optional shapely Polygon, used only if we need to uniform-init
      dtype:         overrides dtype; defaults to projector’s dtype if None

    Returns:
      x: (M,2) tensor on projector.device, inside the domain (up to SDF discretization)
    """
    dev = projector.sdf_grid.device
    dt  = projector.sdf_grid.dtype if dtype is None else dtype

    # --- Initialization (no rasterization here) ---
    if init_x is not None:
        # use provided warm-start
        x = init_x.to(device=dev, dtype=dt)
        if x.shape != (M, 2):
            raise ValueError(f"init_x must be shape (M,2) = ({M},2); got {tuple(x.shape)}")
    else:
        if init_sampler is not None:
            x0 = init_sampler(M)
            x0 = x0.cpu().numpy() if torch.is_tensor(x0) else np.asarray(x0, dtype=np.float32)
            x = torch.from_numpy(x0).to(device=dev, dtype=dt)
        else:
            if poly is None:
                raise ValueError("Need either init_x, init_sampler, or poly for uniform init.")
            # one CPU->GPU hop at start only
            x0 = _uniform_in_polygon(poly, M, device="cpu", dtype=torch.float32).numpy()
            x  = torch.from_numpy(x0).to(device=dev, dtype=dt)

    # make sure any edge cases start inside
    x = projector.project(x, iters=project_iters, push_in=push_in)

    # --- Langevin with occasional projection (all on GPU) ---
    for t in range(T):
        sigma_t = sigma_noise * (alpha ** t)
        s = model(x)                              # (M,2)
        noise = torch.randn_like(x) * sigma_t     # (M,2)
        x = x + 0.5 * (eps ** 2) * s + eps * noise

        if (t % project_every == 0) or (t == T - 1):
            x = projector.project(x, iters=project_iters, push_in=push_in)

    return x

def counts_from_samples_nearest(spots, samples, return_assignments=False, chunk=200_000):
    """
    Assign each sample to its nearest spot and return counts per spot.

    Parameters
    ----------
    spots : array-like, shape (S, D)
        Spot coordinates (e.g., Visium spot centers).
    samples : array-like, shape (N, D)
        Generated transcript coordinates.
    return_assignments : bool, default False
        If True, also return an array of length N with the chosen spot index for each sample.
    chunk : int, default 200_000
        Only used in the NumPy fallback; processes samples in batches to avoid large memory usage.

    Returns
    -------
    counts : np.ndarray, shape (S,), dtype=int
        Number of samples assigned to each spot (in the same order as `spots`).
    (optional) idx : np.ndarray, shape (N,), dtype=int
        The index of the nearest spot for each sample.
    """
    spots = np.asarray(spots, dtype=float)
    samples = np.asarray(samples, dtype=float)

    S = spots.shape[0]
    if S == 0:
        raise ValueError("`spots` is empty.")
    if samples.size == 0:
        counts = np.zeros(S, dtype=np.int64)
        return (counts, np.empty(0, dtype=np.int64)) if return_assignments else counts

    # Try fast path: SciPy cKDTree
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(spots)
        _, idx = tree.query(samples, k=1)  # idx: nearest spot index for each sample
    except Exception:
        # Fallback: pure NumPy, chunked to limit memory
        N = samples.shape[0]
        idx_parts = []
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            block = samples[start:end]                  # (B, D)
            # Pairwise squared distances to spots: (B, S)
            d2 = ((block[:, None, :] - spots[None, :, :]) ** 2).sum(axis=2)
            idx_parts.append(np.argmin(d2, axis=1))
        idx = np.concatenate(idx_parts, axis=0)

    counts = np.bincount(idx, minlength=S).astype(np.int64)
    return (counts, idx) if return_assignments else counts

#Usage:
#projector = build_sdf_projector(poly, H=1024, W=1024, margin=0.03, device="cuda")
# x_samples = sample_ld_in_polygon_sdf(
#         model, projector,
#         M=np.load(os.path.join("spagradient_runs_DLPFC", "panel_500_lognorm",gene, "counts_simulated.npy")).sum(), T=750, eps=0.15,
#         sigma_noise=1.0, alpha=0.999,
#         project_every=1, project_iters=1, push_in=1e-3,
#         poly=poly
#     )
# x_samples = x_samples.cpu().numpy()
# counts_simulated = counts_from_samples_nearest(
#                 coords_t, x_samples, return_assignments=False)

# ─── 7. Plotting Function ───────────────────────────────────────────────────
def plot_score_field(
    model,
    coords_t: torch.Tensor,
    weights_t: torch.Tensor,
    gene: str,
    percentile: float = 100,
    arrow_scale_factor: float = 0.5,
    quiver_scale: float = 5.0,
    cmap: str = 'magma',
    marker: str = 'h',
    figsize: tuple = (15, 8),
):
    """
    Plot spot expression density and learned score field side by side.

    Parameters
    ----------
    model : torch.nn.Module
        Trained score network.
    coords_t : torch.Tensor, shape (N, 2)
        Spatial coordinates of spots (CPU tensor).
    weights_t : torch.Tensor, shape (N,)
        Spot weights or expression values.
    gene : str
        Gene name for title annotation.
    percentile : float, default=95
        Percentile cutoff for arrow magnitudes.
    arrow_scale_factor : float, default=0.5
        Factor to scale U and V components before quiver.
    quiver_scale : float, default=5.0
        The `scale` parameter for plt.quiver.
    cmap : str, default='magma'
        Colormap for scatter points.
    marker : str, default='h'
        Marker style for scatter.
    figsize : tuple, default=(15, 8)
        Figure size.

    Returns
    -------
    fig, axes
        Matplotlib Figure and Axes array.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coords = coords_t.to(device)
    model = model.to(device)
    
    with torch.no_grad():
        scores = model(coords).cpu().numpy()
    
    U, V = scores[:, 0], scores[:, 1]
    mags = np.sqrt(U**2 + V**2)

    # Compute cutoff and mask
    cutoff = np.percentile(mags, percentile)
    mask = mags < cutoff

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    ax1, ax2 = axes

    # 1) Scatter only
    sc1 = ax1.scatter(
        coords_t[:, 0], coords_t[:, 1],
        c=weights_t, cmap=cmap,
        marker=marker, s=60, edgecolor='gray', linewidth=0.2
    )
    ax1.set_title(f"Spot Expression Density: {gene}")
    ax1.set_aspect('equal')
    ax1.axis('off')

    # 2) Scatter + quiver
    sc2 = ax2.scatter(
        coords_t[:, 0], coords_t[:, 1],
        c=weights_t, cmap=cmap,
        marker=marker, s=60, edgecolor='gray', linewidth=0.2
    )
    ax2.quiver(
        coords_t[mask, 0], coords_t[mask, 1],
        U[mask] * arrow_scale_factor, V[mask] * arrow_scale_factor,
        angles='xy', scale_units='xy', scale=quiver_scale,
        color='white', width=0.0025
    )
    ax2.set_title(f"Score Field Overlay: {gene}")
    ax2.set_aspect('equal')
    ax2.axis('off')

    # Shared Colorbar
    cbar = fig.colorbar(sc2, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02)
    cbar.set_label('weights')
    
    return fig, axes 