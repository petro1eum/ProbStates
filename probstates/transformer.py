"""
Transformer-based parameter learner for ProbStates fusion.

Goal: learn to map raw market feature sequences to optimal fusion parameters
(weights and per-feature phase biases or per-mode kappa) that maximize a proxy
objective (e.g., Sharpe of next-period return or classification of next sign),
while keeping ProbStates aggregation (`aggregate_specs`) as the decision core.

Implementation uses PyTorch. If PyTorch is unavailable, raise a clear error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import math
import numpy as np
import os
import time

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
except Exception as e:  # pragma: no cover
    torch = None
    nn = None
    Dataset = object
    DataLoader = object

from probstates.markets import FeatureSpec, aggregate_specs, aggregate_specs_mc, indicator_to_prob
from probstates import set_phase_or_mode


@dataclass
class FusionHeadConfig:
    num_features: int
    mode: str = 'weight'  # ⊕₄ mode
    learn_kappa: bool = False  # if using programmable fusion outside, leave False


class MarketSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        assert X.ndim == 2
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.seq_len = seq_len
        self.n = X.shape[0]

    def __len__(self):
        return max(0, self.n - self.seq_len)

    def __getitem__(self, idx: int):
        xs = self.X[idx: idx + self.seq_len]
        target = self.y[idx + self.seq_len - 1]
        return xs, target


class ProbStatesTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1, fusion_cfg: Optional[FusionHeadConfig] = None, pred_dim: int = 1):
        super().__init__()
        if torch is None:
            raise ImportError("PyTorch is required for ProbStatesTransformer")
        self.fusion_cfg = fusion_cfg or FusionHeadConfig(num_features=input_dim)
        self.input_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        # Head produces per-feature parameters: weight, phase_bias, alpha, beta, mu, kappa_vm
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 6 * self.fusion_cfg.num_features)
        )
        # regression head for next return(s)
        self.pred = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(d_model, pred_dim))
        self.pred_dim = pred_dim
        # Learnable temperature for smooth position
        self.tau = nn.Parameter(torch.tensor(0.02, dtype=torch.float32))
        # Learnable base threshold for position
        self.thr = nn.Parameter(torch.tensor(0.55, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Accept (B, F) or (B, T, F)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.input_proj(x)
        h = self.encoder(h)
        # use last token embedding
        z = h[:, -1, :]
        params = self.head(z)
        pred = self.pred(z)
        if pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
        return params, pred

    def build_specs(self, last_features: np.ndarray) -> List[FeatureSpec]:
        set_phase_or_mode(self.fusion_cfg.mode)
        self.eval()
        # standardize if stats present
        lf = last_features.astype(np.float32)
        if hasattr(self, 'mu') and hasattr(self, 'sigma'):
            mu = self.mu.detach().cpu().numpy()
            sigma = self.sigma.detach().cpu().numpy()
            lf = (lf - mu) / (sigma + 1e-6)
        with torch.no_grad():
            device = next(self.parameters()).device
            x = torch.tensor(lf[None, ...], dtype=torch.float32, device=device)  # (1, F) → forward will add T=1
            params, _ = self.forward(x)
        params = params.cpu().numpy().reshape(-1)
        nf = self.fusion_cfg.num_features
        # unpack per-feature 6-tuple
        params = params.reshape(nf, 6)
        weights = np.clip(params[:, 0], 0.0, 4.0)
        ph_bias = np.clip(params[:, 1], -math.pi, math.pi)
        alpha = np.clip(params[:, 2], 0.0, 50.0)
        beta = np.clip(params[:, 3], 0.0, 50.0)
        mu = np.clip(params[:, 4], -math.pi, math.pi)
        kappa_vm = np.clip(params[:, 5], 0.0, 50.0)
        specs: List[FeatureSpec] = []
        # Interpret last_features as: columns are standardized indicators; map to probability via indicator_to_prob
        probs = indicator_to_prob(last_features)
        for i in range(nf):
            p = float(np.clip(probs[i], 0.0, 1.0))
            phi = float((ph_bias[i] + math.pi) % (2 * math.pi))
            specs.append(FeatureSpec(
                name=f'f{i}', prob=p, phase=phi, weight=float(max(0.5, weights[i])),
                alpha=float(alpha[i]), beta=float(beta[i]), mu=float(mu[i]), kappa_vm=float(kappa_vm[i])
            ))
        return specs


def train_transformer(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = 64,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    lr: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 64,
    mode: str = 'weight',
    weight_decay: float = 1e-4,
    pred_dim: int = 1,
    val_ratio: float = 0.2,
    patience: int = 5,
    fusion_train: bool = False,
    l1_weight: float = 1e-4,
    sharpe_weight: float = 0.1,
    brier_weight: float = 0.0,
    ece_weight: float = 0.0,
    overconf_weight: float = 0.0,
    fee_weight: float = 0.0,
) -> ProbStatesTransformer:
    if torch is None:
        raise ImportError("PyTorch is required for training")
    # Robust standardization (median/IQR) + clipping
    med = np.nanmedian(X, axis=0)
    iqr = np.nanpercentile(X, 75, axis=0) - np.nanpercentile(X, 25, axis=0)
    iqr = np.where(iqr < 1e-6, 1e-6, iqr)
    Xn = (X - med) / iqr
    Xn = np.clip(Xn, -10.0, 10.0)
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
    yn = np.nan_to_num(y, nan=0.0)
    # train/val split
    n = Xn.shape[0]
    split = int(n * (1.0 - val_ratio))
    ds_tr = MarketSequenceDataset(Xn[:split], yn[:split], seq_len)
    ds_va = MarketSequenceDataset(Xn[split:], yn[split:], seq_len)
    workers = max(0, min(8, (os.cpu_count() or 4) - 2))
    dl = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        persistent_workers=(workers > 0),
        prefetch_factor=(2 if workers > 0 else None),
    )
    dlv = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        persistent_workers=(workers > 0),
        prefetch_factor=(2 if workers > 0 else None),
    )
    model = ProbStatesTransformer(input_dim=Xn.shape[1], d_model=d_model, nhead=nhead, num_layers=num_layers, fusion_cfg=FusionHeadConfig(num_features=Xn.shape[1], mode=mode), pred_dim=pred_dim)
    # persist normalization stats
    model.mu = torch.tensor(med, dtype=torch.float32)
    model.sigma = torch.tensor(iqr, dtype=torch.float32)
    # Prefer Apple Metal (MPS) on macOS, then CUDA, else CPU
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    model.to(device)
    print(f"[Trainer] device={device}, workers={workers}, X={Xn.shape}, seq_len={seq_len}, batches={len(dl)}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)
    loss_mse = nn.MSELoss()
    loss_bce = nn.BCELoss()

    def _vm_A(kappa: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        small = kappa <= 1.0
        out = torch.empty_like(kappa)
        out[small] = kappa[small] / 2.0
        large = ~small
        if large.any():
            k = torch.clamp(kappa[large], min=1.0 + eps)
            out[large] = 1.0 - 1.0 / (2.0 * k) - 1.0 / (8.0 * k * k)
        return torch.clamp(out, 0.0, 1.0)
    best_val = float('inf'); bad = 0; best_state = None
    checkpoints: List[Dict[str, torch.Tensor]] = []
    def _ece_metric(p: torch.Tensor, t: torch.Tensor, bins: int = 10) -> torch.Tensor:
        edges = torch.linspace(0, 1, bins + 1, device=p.device)
        ece = torch.tensor(0.0, device=p.device)
        for i in range(bins):
            m = (p >= edges[i]) & (p < edges[i + 1])
            if m.any():
                avg_p = torch.mean(p[m])
                freq = torch.mean(t[m])
                ece = ece + torch.abs(avg_p - freq) * (m.float().mean())
        return ece
    for epoch in range(epochs):
        model.train()
        total = 0.0
        tr_bce = tr_brier = tr_ece = tr_sharpe = tr_over = tr_fee = tr_reg = 0.0
        t0 = time.time()
        # Warmup for tau: freeze for first 3 epochs
        model.tau.requires_grad = (epoch >= 3)
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            if pred_dim > 1:
                yb = yb.view(-1, 1).repeat(1, pred_dim)
            else:
                if yb.dim() > 1:
                    yb = yb.view(-1)
            opt.zero_grad()
            params, pred = model(xb)
            # unpack per-feature params
            nf = model.fusion_cfg.num_features
            params = params.view(-1, nf, 6)
            weights = torch.clamp(params[:, :, 0], 0.0, 4.0)
            phase_bias = torch.clamp(params[:, :, 1], -math.pi, math.pi)
            alpha = torch.clamp(params[:, :, 2], 0.0, 50.0)
            beta = torch.clamp(params[:, :, 3], 0.0, 50.0)
            mu = torch.clamp(params[:, :, 4] + phase_bias, -math.pi, math.pi)
            kappa_vm = torch.clamp(params[:, :, 5], 0.0, 50.0)

            if fusion_train:
                last_x = xb[:, -1, :]  # (B, F)
                # Expected probability per feature: prefer Beta mean if α,β>0 else sigmoid(feature)
                p_mean_base = torch.sigmoid(last_x)
                alpha_pos = torch.clamp(alpha, min=0.0)
                beta_pos = torch.clamp(beta, min=0.0)
                denom = alpha_pos + beta_pos + 1e-6
                p_beta = torch.where(denom > 1e-6, alpha_pos / denom, p_mean_base)
                use_beta = (alpha_pos > 0.1) & (beta_pos > 0.1)
                p_mean = torch.where(use_beta, p_beta, p_mean_base)
                # Phase concentration via von Mises A(kappa)
                amp = torch.sqrt(torch.clamp(p_mean, 0.0, 1.0)) * _vm_A(kappa_vm)
                re = torch.sum(weights * amp * torch.cos(mu), dim=1)
                im = torch.sum(weights * amp * torch.sin(mu), dim=1)
                p_agg = torch.clamp(re * re + im * im, 0.0, 1.0)
                target = (yb > 0.0).float()
                loss_bce_core = loss_bce(p_agg, target)
                loss_brier = torch.mean((p_agg - target) ** 2)
                # ECE (Expected Calibration Error) with 10 bins
                if ece_weight > 0.0:
                    bins = torch.linspace(0, 1, 11, device=xb.device)
                    ece = torch.tensor(0.0, device=xb.device)
                    for i in range(10):
                        m = (p_agg >= bins[i]) & (p_agg < bins[i+1])
                        if m.any():
                            avg_p = torch.mean(p_agg[m])
                            freq = torch.mean(target[m])
                            ece = ece + torch.abs(avg_p - freq) * (m.float().mean())
                else:
                    ece = torch.tensor(0.0, device=xb.device)
                # Sharpe surrogate with smooth position
                # Warmup: freeze tau for first epochs (handled outside via epoch counter)
                thr = torch.clamp(model.thr, 0.5, 0.6)
                pos_smooth = torch.sigmoid((p_agg - thr) / torch.clamp(model.tau, min=1e-3))
                ret_proxy = pos_smooth * yb
                mean_ret = torch.mean(ret_proxy)
                std_ret = torch.sqrt(torch.var(ret_proxy) + 1e-8)
                loss_sharpe = - mean_ret / (std_ret + 1e-8)
                # Overconfidence penalty (confident wrong predictions)
                over_pen = torch.relu(p_agg - 0.8) * (1.0 - target) + torch.relu((1.0 - p_agg) - 0.8) * target
                loss_over = torch.mean(over_pen)
                # Turnover surrogate: approximate previous position from sequence t-1
                if xb.size(1) >= 2:
                    x_prev = xb[:, :-1, :]
                    params_prev, _ = model(x_prev)
                    nf_prev = model.fusion_cfg.num_features
                    params_prev = params_prev.view(-1, nf_prev, 6)
                    w_p = torch.clamp(params_prev[:, :, 0], 0.0, 4.0)
                    phb_p = torch.clamp(params_prev[:, :, 1], -math.pi, math.pi)
                    a_p = torch.clamp(params_prev[:, :, 2], 0.0, 50.0)
                    b_p = torch.clamp(params_prev[:, :, 3], 0.0, 50.0)
                    mu_p = torch.clamp(params_prev[:, :, 4] + phb_p, -math.pi, math.pi)
                    kvm_p = torch.clamp(params_prev[:, :, 5], 0.0, 50.0)
                    last_prev = x_prev[:, -1, :]
                    p_mean_base_p = torch.sigmoid(last_prev)
                    denom_p = a_p + b_p + 1e-6
                    p_beta_p = torch.where(denom_p > 1e-6, a_p / denom_p, p_mean_base_p)
                    use_beta_p = (a_p > 0.1) & (b_p > 0.1)
                    p_mean_p = torch.where(use_beta_p, p_beta_p, p_mean_base_p)
                    amp_p = torch.sqrt(torch.clamp(p_mean_p, 0.0, 1.0)) * _vm_A(kvm_p)
                    re_p = torch.sum(w_p * amp_p * torch.cos(mu_p), dim=1)
                    im_p = torch.sum(w_p * amp_p * torch.sin(mu_p), dim=1)
                    p_agg_prev = torch.clamp(re_p * re_p + im_p * im_p, 0.0, 1.0)
                    pos_prev = torch.sigmoid((p_agg_prev - thr) / torch.clamp(model.tau, min=1e-3))
                    loss_fee = torch.mean(torch.abs(pos_smooth - pos_prev))
                else:
                    loss_fee = torch.tensor(0.0, device=xb.device)
                loss_core = (
                    loss_bce_core
                    + brier_weight * loss_brier
                    + ece_weight * ece
                    + sharpe_weight * loss_sharpe
                    + overconf_weight * loss_over
                    + fee_weight * loss_fee
                )
            else:
                loss_core = loss_mse(pred, yb)

            reg_conf = 1e-4 * (alpha.mean() + beta.mean() + kappa_vm.mean())
            reg_l1 = l1_weight * torch.mean(torch.abs(weights))
            loss = loss_core + reg_conf + reg_l1
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += float(loss.item())
            if fusion_train:
                tr_reg += float((reg_conf + reg_l1).item())
                tr_bce += float(loss_bce_core.item())
                tr_brier += float(loss_brier.item())
                # recompute ece only if enabled
                if ece_weight > 0.0:
                    tr_ece += float(_ece_metric(p_agg, target).item())
                tr_sharpe += float(loss_sharpe.item())
                tr_over += float(loss_over.item())
                tr_fee += float(loss_fee.item())
        # validation
        model.eval(); vloss = 0.0; vc = 0
        v_bce_sum = v_brier_sum = v_ece_sum = v_hit_n = v_hit_d = 0.0
        with torch.no_grad():
            for xb, yb in dlv:
                xb = xb.to(device)
                yb = yb.to(device)
                if pred_dim > 1:
                    yb = yb.view(-1, 1).repeat(1, pred_dim)
                else:
                    if yb.dim() > 1:
                        yb = yb.view(-1)
                params, pred = model(xb)
                if fusion_train:
                    nf = model.fusion_cfg.num_features
                    params = params.view(-1, nf, 6)
                    weights = torch.clamp(params[:, :, 0], 0.0, 4.0)
                    phase_bias = torch.clamp(params[:, :, 1], -math.pi, math.pi)
                    alpha = torch.clamp(params[:, :, 2], 0.0, 50.0)
                    beta = torch.clamp(params[:, :, 3], 0.0, 50.0)
                    mu = torch.clamp(params[:, :, 4] + phase_bias, -math.pi, math.pi)
                    kappa_vm = torch.clamp(params[:, :, 5], 0.0, 50.0)
                    last_x = xb[:, -1, :]
                    p_mean_base = torch.sigmoid(last_x)
                    denom = alpha + beta + 1e-6
                    p_beta = torch.where(denom > 1e-6, alpha / denom, p_mean_base)
                    use_beta = (alpha > 0.1) & (beta > 0.1)
                    p_mean = torch.where(use_beta, p_beta, p_mean_base)
                    amp = torch.sqrt(torch.clamp(p_mean, 0.0, 1.0)) * _vm_A(kappa_vm)
                    re = torch.sum(weights * amp * torch.cos(mu), dim=1)
                    im = torch.sum(weights * amp * torch.sin(mu), dim=1)
                    p_agg = torch.clamp(re * re + im * im, 0.0, 1.0)
                    target = (yb > 0.0).float()
                    v_bce = loss_bce(p_agg, target)
                    v_brier = torch.mean((p_agg - target) ** 2)
                    ece = _ece_metric(p_agg, target) if ece_weight > 0.0 else torch.tensor(0.0, device=xb.device)
                    vloss += float((v_bce + brier_weight * v_brier + ece_weight * ece).item()); vc += 1
                    v_bce_sum += float(v_bce.item()); v_brier_sum += float(v_brier.item()); v_ece_sum += float(ece.item())
                    # hit-rate at 0.5
                    pred_cls = (p_agg >= 0.5).float()
                    v_hit_n += float(torch.mean((pred_cls == target).float()).item()); v_hit_d += 1.0
                else:
                    vloss += float(loss_mse(pred, yb).item()); vc += 1
        vloss = vloss / max(vc, 1)
        mean_train = total / max(len(dl), 1)
        dt = time.time() - t0
        lr = opt.param_groups[0]['lr']
        if fusion_train:
            nb = max(len(dl), 1)
            log_tr = f"bce={tr_bce/nb:.4f} brier={tr_brier/nb:.4f} ece={(tr_ece/nb) if ece_weight>0 else 0.0:.4f} sharpe={tr_sharpe/nb:.4f} over={tr_over/nb:.4f} fee={tr_fee/nb:.4f} reg={tr_reg/nb:.4f}"
            log_va = f"vbce={v_bce_sum/max(vc,1):.4f} vbrier={v_brier_sum/max(vc,1):.4f} vece={(v_ece_sum/max(vc,1)) if ece_weight>0 else 0.0:.4f} vhit={(v_hit_n/max(v_hit_d,1.0)):.3f}"
        else:
            log_tr = "mse"; log_va = "vmse"
        print(f"[Trainer] epoch {epoch+1}/{epochs} time={dt:.1f}s lr={lr:.2e} train_loss={mean_train:.4f} val_loss={vloss:.4f} {log_tr} | {log_va} early_stop={bad}/{patience}", flush=True)
        if vloss < best_val:
            best_val = vloss; bad = 0; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            # Save up to 3 best checkpoints
            checkpoints.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            if len(checkpoints) > 3:
                checkpoints.pop(0)
        else:
            bad += 1
            if bad >= patience:
                print(f"[Trainer] early stopping at epoch {epoch+1}", flush=True)
                break
        # LR scheduler step
        scheduler.step(vloss)
    if checkpoints:
        # Average checkpoints
        keys = list(checkpoints[0].keys())
        avg_state = {}
        for k in keys:
            avg_state[k] = torch.mean(torch.stack([cp[k] for cp in checkpoints]), dim=0)
        model.load_state_dict(avg_state)
    elif best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_with_model(model: ProbStatesTransformer, X: np.ndarray, seq_len: int) -> Tuple[np.ndarray, List[List[FeatureSpec]]]:
    if torch is None:
        raise ImportError("PyTorch is required")
    # apply same standardization approach as in training (assume already standardized upstream for simplicity)
    Xf = X.astype(np.float32)
    # standardize using model stats if present
    if hasattr(model, 'mu') and hasattr(model, 'sigma'):
        mu = model.mu.detach().cpu().numpy()
        sigma = model.sigma.detach().cpu().numpy()
        Xf = (Xf - mu) / (sigma + 1e-6)
    Xn = np.nan_to_num(Xf, nan=0.0, posinf=0.0, neginf=0.0)
    specs_list: List[List[FeatureSpec]] = []
    preds: List[float] = []
    model.eval()
    with torch.no_grad():
        for i in range(seq_len, Xn.shape[0] + 1):
            last_seq = Xn[i - seq_len: i]
            device = next(model.parameters()).device
            x = torch.tensor(last_seq[None, ...], dtype=torch.float32, device=device)
            params, pred = model(x)
            preds.append(float(pred.item()))
            nf = model.fusion_cfg.num_features
            params_np = params.cpu().numpy().reshape(-1)
            # Parse exactly as in build_specs: (nf, 6) → weight, phase_bias, alpha, beta, mu, kappa_vm
            params_np = params_np.reshape(nf, 6)
            weights = np.clip(params_np[:, 0], 0.0, 4.0)
            ph_bias = np.clip(params_np[:, 1], -math.pi, math.pi)
            alpha = np.clip(params_np[:, 2], 0.0, 50.0)
            beta = np.clip(params_np[:, 3], 0.0, 50.0)
            mu = np.clip(params_np[:, 4], -math.pi, math.pi)
            kappa_vm = np.clip(params_np[:, 5], 0.0, 50.0)
            last_features = last_seq[-1]
            probs = indicator_to_prob(last_features)
            specs: List[FeatureSpec] = []
            for j in range(nf):
                p = float(np.clip(probs[j], 0.0, 1.0))
                phi = float((ph_bias[j] + math.pi) % (2 * math.pi))
                specs.append(FeatureSpec(
                    name=f'f{j}',
                    prob=p,
                    phase=phi,
                    weight=float(max(0.5, weights[j])),
                    alpha=float(alpha[j]),
                    beta=float(beta[j]),
                    mu=float(mu[j]),
                    kappa_vm=float(kappa_vm[j]),
                ))
            specs_list.append(specs)
    return np.array(preds), specs_list


def save_model(model: ProbStatesTransformer, path: str) -> None:
    if torch is None:
        raise ImportError("PyTorch is required")
    payload = {
        'state_dict': model.state_dict(),
        'cfg': {
            'input_dim': model.input_proj.in_features,
            'd_model': model.input_proj.out_features,
            'nhead': model.encoder.layers[0].self_attn.num_heads if hasattr(model.encoder.layers[0].self_attn, 'num_heads') else 4,
            'num_layers': len(model.encoder.layers),
            'mode': model.fusion_cfg.mode,
            'num_features': model.fusion_cfg.num_features,
        },
        'mu': getattr(model, 'mu', None),
        'sigma': getattr(model, 'sigma', None),
    }
    # detach tensors for CPU save
    if isinstance(payload['mu'], torch.Tensor):
        payload['mu'] = payload['mu'].detach().cpu()
    if isinstance(payload['sigma'], torch.Tensor):
        payload['sigma'] = payload['sigma'].detach().cpu()
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_model(path: str) -> ProbStatesTransformer:
    if torch is None:
        raise ImportError("PyTorch is required")
    ckpt = torch.load(path, map_location='cpu')
    cfg = ckpt['cfg']
    model = ProbStatesTransformer(
        input_dim=cfg['input_dim'],
        d_model=cfg['d_model'],
        nhead=cfg.get('nhead', 4),
        num_layers=cfg.get('num_layers', 2),
        fusion_cfg=FusionHeadConfig(num_features=cfg['num_features'], mode=cfg.get('mode','weight')),
    )
    model.load_state_dict(ckpt['state_dict'])
    if ckpt.get('mu') is not None:
        model.mu = ckpt['mu'] if isinstance(ckpt['mu'], torch.Tensor) else torch.tensor(ckpt['mu'])
    if ckpt.get('sigma') is not None:
        model.sigma = ckpt['sigma'] if isinstance(ckpt['sigma'], torch.Tensor) else torch.tensor(ckpt['sigma'])
    model.eval()
    return model


