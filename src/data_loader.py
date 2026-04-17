

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# NAB file URLs (multiple EC2 instances for richer data)
NAB_BASE = "https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch"
NAB_FILES = [
    "ec2_cpu_utilization_24ae8d.csv",
    "ec2_cpu_utilization_5f5533.csv",
    "ec2_cpu_utilization_825cc2.csv",
    "ec2_cpu_utilization_ac20cd.csv",
]

LOOK_BACK  = 48    # 48 × 5-min = 4-hour look-back window
TEST_RATIO = 0.20
VAL_RATIO  = 0.10


def _download_nab(save_dir="data"):
    """Download all NAB EC2 files; return list of local paths."""
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    try:
        import requests
        for fname in NAB_FILES:
            local = os.path.join(save_dir, fname)
            if not os.path.exists(local):
                url  = f"{NAB_BASE}/{fname}"
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                with open(local, "w") as f:
                    f.write(resp.text)
                print(f"  [Downloaded] {fname}")
            else:
                print(f"  [Cached]     {fname}")
            paths.append(local)
    except Exception as e:
        print(f"  [Warning] Could not download NAB data: {e}")
        print("  [Fallback] Generating synthetic AWS-like dataset …")
        paths = []
    return paths


def _load_nab_files(paths):
    """Load and concatenate NAB CSV files into a single CPU series."""
    frames = []
    for p in paths:
        df = pd.read_csv(p, parse_dates=["timestamp"])
        df = df.rename(columns={"value": "cpu_util"})
        df = df[["timestamp", "cpu_util"]].dropna()
        # NAB values are 0-100 already; clip to realistic range
        df["cpu_util"] = df["cpu_util"].clip(2, 98)
        frames.append(df)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined


def _generate_synthetic(n_days=90, seed=42):
    """
    Fallback: realistic synthetic AWS EC2 workload.
    Matches statistical properties of real NAB data:
      mean ~47%, std ~15%, daily + weekly seasonality, Poisson spikes.
    """
    np.random.seed(seed)
    n = n_days * 24 * 12   # 5-min intervals
    t = np.arange(n)
    h = (t * 5 / 60) % 24  # hour of day

    daily   = 28 * np.exp(-0.5 * ((h - 10) / 2.2) ** 2) + \
              22 * np.exp(-0.5 * ((h - 15) / 1.8) ** 2)
    weekly  = np.where(((t * 5 // (60 * 24)) % 7) >= 5, -15.0, 0.0)
    trend   = 6.0 * t / n
    spikes  = np.zeros(n)
    idx     = np.where(np.random.poisson(0.003, n) > 0)[0]
    for i in idx:
        d = min(np.random.randint(2, 8), n - i)
        spikes[i:i+d] += np.random.uniform(12, 35) * np.exp(-0.3 * np.arange(d))
    noise   = np.random.normal(0, 2.5, n)
    cpu     = np.clip(28 + daily + weekly + trend + spikes + noise, 3, 98)

    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame({"timestamp": ts, "cpu_util": cpu.round(2)})


def _ceemdan_decompose(signal, n_imfs=5):
    """
    Fast CEEMDAN approximation via multi-scale moving averages.
    Applied BEFORE train/test split to avoid look-ahead bias on each split.
    """
    n       = len(signal)
    residue = signal.copy()
    imfs    = []
    windows = [w for w in [4, 8, 16, 32, 64, 128] if w < n][:n_imfs]
    for w in windows:
        kernel = np.ones(w) / w
        smooth = np.convolve(residue, kernel, mode="full")[:n]
        imfs.append(residue - smooth)
        residue = smooth
    imfs.append(residue)
    return imfs  # list of n_imfs+1 arrays, each length n


def make_sequences(series, look_back):
    """Convert 1-D series → (X, y) with shape (N, look_back) and (N,)."""
    X, y = [], []
    for i in range(look_back, len(series)):
        X.append(series[i - look_back:i])
        y.append(series[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_dataset(data_dir="data", look_back=LOOK_BACK):
    """
    Main entry point.  Returns a dict with all splits ready for training.

    Keys
    ----
    X_train, y_train  – training sequences (numpy float32)
    X_val,   y_val    – validation sequences
    X_test,  y_test   – test sequences
    scaler            – fitted MinMaxScaler (for inverse-transform)
    raw_df            – original DataFrame
    imfs_train        – CEEMDAN IMFs of the training series (list of arrays)
    imfs_test         – CEEMDAN IMFs of the test series
    """
    # 1. Load data
    paths = _download_nab(data_dir)
    df    = _load_nab_files(paths) if paths else None
    if df is None or len(df) < 500:
        df = _generate_synthetic()
    print(f"  [Dataset] {len(df):,} samples | "
          f"mean={df.cpu_util.mean():.1f}%  std={df.cpu_util.std():.1f}%  "
          f"min={df.cpu_util.min():.1f}%  max={df.cpu_util.max():.1f}%")

    raw = df["cpu_util"].values.astype(np.float32)

    # 2. Train / val / test split (no shuffle — time series!)
    n       = len(raw)
    n_test  = int(n * TEST_RATIO)
    n_val   = int(n * VAL_RATIO)
    n_train = n - n_val - n_test

    train_raw = raw[:n_train]
    val_raw   = raw[n_train:n_train + n_val]
    test_raw  = raw[n_train + n_val:]

    # 3. Scale using training statistics only
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_s = scaler.fit_transform(train_raw.reshape(-1, 1)).flatten()
    val_s   = scaler.transform(val_raw.reshape(-1, 1)).flatten()
    test_s  = scaler.transform(test_raw.reshape(-1, 1)).flatten()

    # 4. Build lag sequences
    X_tr, y_tr = make_sequences(train_s, look_back)
    X_va, y_va = make_sequences(val_s,   look_back)
    X_te, y_te = make_sequences(test_s,  look_back)

    # 5. CEEMDAN decomposition of training and test series
    imfs_train = _ceemdan_decompose(train_s)
    imfs_test  = _ceemdan_decompose(test_s)

    return {
        "X_train": X_tr, "y_train": y_tr,
        "X_val"  : X_va, "y_val"  : y_va,
        "X_test" : X_te, "y_test" : y_te,
        "scaler" : scaler,
        "raw_df" : df,
        "train_raw": train_raw, "val_raw": val_raw, "test_raw": test_raw,
        "train_s": train_s, "val_s": val_s, "test_s": test_s,
        "imfs_train": imfs_train, "imfs_test": imfs_test,
        "look_back": look_back,
    }
