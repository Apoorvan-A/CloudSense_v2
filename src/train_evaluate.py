"""
train_evaluate.py
=================
Unified training and evaluation pipeline using PyTorch.

Features
--------
- GPU support (uses CUDA if available, else CPU)
- Early stopping (patience=15 epochs)
- Learning rate scheduler (ReduceLROnPlateau)
- Proper train / val / test split with no data leakage
- sklearn metrics for clean evaluation
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#  Metrics
def mape(y_true, y_pred, eps=1e-8):
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def compute_metrics(y_true, y_pred):
    return {
        "MAE" : round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "MAPE": round(mape(y_true, y_pred), 4),
        "R2"  : round(float(r2_score(y_true, y_pred)), 4),
    }


#  Generic trainer for standard models (LSTM, CNN-LSTM, BiLSTM, Transformer)
def train_model(model, data, epochs=150, lr=1e-3, batch_size=64,
                patience=15, verbose=True):
    """
    Train a standard model (non-CEEMDAN) with early stopping.

    Parameters
    ----------
    model     : nn.Module
    data      : dict from data_loader.load_dataset()
    epochs    : int
    lr        : float
    batch_size: int
    patience  : int  – early stopping patience

    Returns
    -------
    model     : trained nn.Module
    history   : dict  { 'train_loss': [...], 'val_loss': [...] }
    """
    model = model.to(DEVICE)

    # Build DataLoaders
    X_tr = torch.FloatTensor(data["X_train"]).unsqueeze(-1)  # (N, T, 1)
    y_tr = torch.FloatTensor(data["y_train"]).unsqueeze(-1)
    X_va = torch.FloatTensor(data["X_val"]).unsqueeze(-1)
    y_va = torch.FloatTensor(data["y_val"]).unsqueeze(-1)

    train_dl = DataLoader(TensorDataset(X_tr, y_tr),
                          batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(TensorDataset(X_va, y_va),
                          batch_size=batch_size * 2)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=7)
    criterion = nn.MSELoss()

    history         = {"train_loss": [], "val_loss": []}
    best_val_loss   = float("inf")
    best_state      = None
    epochs_no_improv= 0

    for ep in range(1, epochs + 1):
        # Train
        model.train()
        t_losses = []
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            t_losses.append(loss.item())

        # Validate
        model.eval()
        v_losses = []
        with torch.no_grad():
            for Xb, yb in val_dl:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                v_losses.append(criterion(model(Xb), yb).item())

        t_loss = np.mean(t_losses)
        v_loss = np.mean(v_losses)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss    = v_loss
            best_state       = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
            epochs_no_improv = 0
        else:
            epochs_no_improv += 1

        if verbose and ep % 20 == 0:
            print(f"    Epoch {ep:3d} | train={t_loss:.5f}  val={v_loss:.5f}")

        if epochs_no_improv >= patience:
            if verbose:
                print(f"    Early stopping at epoch {ep}")
            break

    model.load_state_dict(best_state)
    return model, history


def evaluate_model(model, data, scaler):
    """Run model on test set; return inverse-scaled predictions and metrics."""
    model.eval()
    X_te = torch.FloatTensor(data["X_test"]).unsqueeze(-1).to(DEVICE)
    with torch.no_grad():
        preds_s = model(X_te).cpu().numpy().flatten()

    preds_s = np.clip(preds_s, 0, 1)
    preds   = scaler.inverse_transform(preds_s.reshape(-1, 1)).flatten()
    y_true  = scaler.inverse_transform(
        data["y_test"].reshape(-1, 1)).flatten()

    return preds, y_true, compute_metrics(y_true, preds)


#  CEEMDAN ensemble trainer
def _build_imf_loaders(imf, look_back, n_train, n_val, batch_size):
    """Build DataLoaders for a single IMF component."""
    from data_loader import make_sequences

    train_imf = imf[:n_train]
    val_imf   = imf[n_train:n_train + n_val]
    test_imf  = imf[n_train + n_val:]

    X_tr, y_tr = make_sequences(train_imf.astype(np.float32), look_back)
    X_va, y_va = make_sequences(val_imf.astype(np.float32),   look_back)
    X_te, y_te = make_sequences(test_imf.astype(np.float32),  look_back)

    def _dl(X, y, shuffle):
        Xt = torch.FloatTensor(X).unsqueeze(-1)
        yt = torch.FloatTensor(y).unsqueeze(-1)
        return DataLoader(TensorDataset(Xt, yt),
                          batch_size=batch_size, shuffle=shuffle)

    return _dl(X_tr, y_tr, True), _dl(X_va, y_va, False), X_te, y_te


def train_ceemdan_model(model, data, epochs=150, lr=1e-3,
                         batch_size=64, patience=15, verbose=True):
    """
    Train CEEMDANBiLSTM by training each IMF sub-model independently.

    1. Decompose full (train+val+test) normalised series into K IMFs.
    2. For each IMF, train the corresponding sub-model on training split.
    3. At evaluation, run each sub-model on its test IMF and sum outputs.
    """
    from data_loader import make_sequences

    model   = model.to(DEVICE)
    full_s  = np.concatenate([data["train_s"], data["val_s"], data["test_s"]])
    n_train = len(data["train_s"])
    n_val   = len(data["val_s"])
    look_back = data["look_back"]
    n_imfs    = model.n_imfs

    # Re-decompose the FULL series for consistency
    from data_loader import _ceemdan_decompose
    imfs = _ceemdan_decompose(full_s, n_imfs=n_imfs - 1)  # returns n_imfs arrays

    # Pad/trim to exactly n_imfs
    while len(imfs) < n_imfs:
        imfs.append(np.zeros_like(imfs[0]))
    imfs = imfs[:n_imfs]

    criterion = nn.MSELoss()
    all_histories = []

    for k, (sub, imf) in enumerate(zip(model.sub_models, imfs)):
        if verbose:
            print(f"    Training IMF sub-model {k+1}/{n_imfs} …", flush=True)

        sub = sub.to(DEVICE)
        tr_dl, va_dl, X_te_k, y_te_k = _build_imf_loaders(
            imf, look_back, n_train, n_val, batch_size)

        opt = torch.optim.Adam(sub.parameters(), lr=lr, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5)

        best_v, best_st, no_improv, hist = float("inf"), None, 0, []
        for ep in range(1, epochs + 1):
            sub.train()
            for Xb, yb in tr_dl:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                criterion(sub(Xb), yb).backward()
                nn.utils.clip_grad_norm_(sub.parameters(), 1.0)
                opt.step()

            sub.eval()
            v_losses = []
            with torch.no_grad():
                for Xb, yb in va_dl:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    v_losses.append(criterion(sub(Xb), yb).item())
            vl = np.mean(v_losses)
            sch.step(vl)
            hist.append(vl)
            if vl < best_v:
                best_v  = vl
                best_st = {k2: v.cpu().clone() for k2, v in sub.state_dict().items()}
                no_improv = 0
            else:
                no_improv += 1
            if no_improv >= patience:
                break

        sub.load_state_dict(best_st)
        all_histories.append(hist)

    return model, all_histories


def evaluate_ceemdan_model(model, data, scaler):
    """Sum per-IMF sub-model predictions to get final forecast."""
    from data_loader import make_sequences, _ceemdan_decompose

    full_s  = np.concatenate([data["train_s"], data["val_s"], data["test_s"]])
    n_train = len(data["train_s"])
    n_val   = len(data["val_s"])
    look_back = data["look_back"]

    imfs = _ceemdan_decompose(full_s, n_imfs=model.n_imfs - 1)
    while len(imfs) < model.n_imfs:
        imfs.append(np.zeros_like(imfs[0]))
    imfs = imfs[:model.n_imfs]

    model.eval()
    preds_sum = None
    y_true_s  = None

    for k, (sub, imf) in enumerate(zip(model.sub_models, imfs)):
        test_imf = imf[n_train + n_val:]
        X_te, y_te = make_sequences(test_imf.astype(np.float32), look_back)
        if len(X_te) == 0:
            continue

        X_t = torch.FloatTensor(X_te).unsqueeze(-1).to(DEVICE)
        with torch.no_grad():
            pred_k = sub(X_t).cpu().numpy().flatten()

        if preds_sum is None:
            preds_sum = pred_k.copy()
            y_true_s  = y_te.copy()
        else:
            mn = min(len(preds_sum), len(pred_k))
            preds_sum[:mn] += pred_k[:mn]
            y_true_s[:mn]  += y_te[:mn]

    if preds_sum is None:
        return np.array([]), np.array([]), {}

    preds_sum = np.clip(preds_sum, 0, 1)
    preds     = scaler.inverse_transform(preds_sum.reshape(-1, 1)).flatten()
    y_true    = scaler.inverse_transform(y_true_s.reshape(-1, 1)).flatten()
    return preds, y_true, compute_metrics(y_true, preds)
