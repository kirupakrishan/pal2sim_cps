import pickle
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import matthews_corrcoef
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
DS_FACTOR    = 25
SEQ_LEN      = 320
STRIDE_TRAIN = 40
STRIDE_TEST  = 20

SENSOR_COLS  = ['Acc.x','Acc.y','Acc.z','Gyro.x','Gyro.y','Gyro.z','Baro.x']
DERIVED_COLS = ['acc_mag','gyro_mag','acc_jerk']
ALL_COLS     = SENSOR_COLS + DERIVED_COLS

SUPERCLASS_MAPPING = {
    "Driving(curve)":                   "Driving(curve)",
    "Driving(straight)":                "Driving(straight)",
    "Lifting(lowering)":                "Lifting(lowering)",
    "Lifting(raising)":                 "Lifting(raising)",
    "Wrapping":                         "Turntable wrapping",
    "Wrapping(preparation)":            "Stationary processes",
    "Docking":                          "Stationary processes",
    "Forks(entering or leaving front)": "Stationary processes",
    "Forks(entering or leaving side)":  "Stationary processes",
    "Standing":                         "Stationary processes",
}
SUPERCLASSES = sorted(set(SUPERCLASS_MAPPING.values()))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ── Preprocessing ─────────────────────────────────────────────────────────────
def apply_superclass(df):
    df = df.copy()
    for sn in SUPERCLASSES:
        children = [k for k,v in SUPERCLASS_MAPPING.items() if v == sn]
        existing = [c for c in children if c in df.columns]
        df[sn] = df[existing].max(axis=1) if existing else 0
    drop = [k for k in SUPERCLASS_MAPPING if k in df.columns and k not in SUPERCLASSES]
    df.drop(columns=drop, inplace=True)
    return df

def downsample_df(df, factor):
    n = (len(df) // factor) * factor
    df = df.iloc[:n]
    sensor_ds = df[SENSOR_COLS].values.reshape(-1, factor, len(SENSOR_COLS)).mean(axis=1)
    label_ds  = df[SUPERCLASSES].values.reshape(-1, factor, len(SUPERCLASSES)).max(axis=1)
    result = pd.DataFrame(sensor_ds, columns=SENSOR_COLS)
    for i, sc in enumerate(SUPERCLASSES):
        result[sc] = label_ds[:, i].astype(int)
    return result

def add_derived(df):
    df = df.copy()
    df['acc_mag']  = np.sqrt(df['Acc.x']**2 + df['Acc.y']**2 + df['Acc.z']**2)
    df['gyro_mag'] = np.sqrt(df['Gyro.x']**2 + df['Gyro.y']**2 + df['Gyro.z']**2)
    df['acc_jerk'] = df['acc_mag'].diff().fillna(0)
    return df

def prepare_experiment(row):
    df = row['data'].copy()
    df.drop(columns=['transportation','container','No loading'], errors='ignore', inplace=True)
    df = apply_superclass(df)
    df = downsample_df(df, DS_FACTOR)
    df = add_derived(df)
    return df

def make_windows(df, scaler=None, fit_scaler=False, stride=1):
    df = df.copy()
    if fit_scaler:
        scaler = RobustScaler()
        df[ALL_COLS] = scaler.fit_transform(df[ALL_COLS])
    else:
        df[ALL_COLS] = scaler.transform(df[ALL_COLS])
    starts = np.arange(0, len(df) - SEQ_LEN + 1, stride)
    X = np.stack([df[ALL_COLS].values[s:s+SEQ_LEN] for s in starts]).astype(np.float32)
    y = df[SUPERCLASSES].values[starts + SEQ_LEN - 1].astype(np.float32)
    return X, y, scaler

def augment_minority(X, y, factor=4):
    mask = ((y[:, SUPERCLASSES.index('Lifting(lowering)')] == 1) |
            (y[:, SUPERCLASSES.index('Lifting(raising)')] == 1))
    Xm, ym = X[mask], y[mask]
    if len(Xm) == 0: return X, y
    Xa, ya = [X], [y]
    for _ in range(factor - 1):
        noise = np.random.normal(0, 0.02, Xm.shape).astype(np.float32)
        scale = np.random.uniform(0.88, 1.12, (Xm.shape[0],1,Xm.shape[2])).astype(np.float32)
        roll  = np.random.randint(-30, 30, Xm.shape[0])
        Xn = Xm * scale + noise
        for i, s in enumerate(roll):
            if s: Xn[i] = np.roll(Xn[i], s, axis=0)
        Xa.append(Xn); ya.append(ym)
    return np.concatenate(Xa), np.concatenate(ya)

def compute_pos_weights(y):
    pos = y.sum(axis=0) + 1e-6
    return torch.tensor(np.clip((len(y)-pos)/pos, 1.0, 10.0),
                        dtype=torch.float32).to(DEVICE)

def get_probs(model, X, batch_size=512):
    model.eval()
    out = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.tensor(X)), batch_size=batch_size):
            out.append(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
    return np.vstack(out)

def find_best_thresholds(probs, y):
    thresholds = []
    print("  Threshold tuning:")
    for i, sc in enumerate(SUPERCLASSES):
        best_t, best_mcc = 0.5, -1
        for t in np.arange(0.05, 0.95, 0.05):
            mcc = matthews_corrcoef(y[:, i], (probs[:, i] > t).astype(int))
            if mcc > best_mcc: best_mcc, best_t = mcc, t
        thresholds.append(best_t)
        print(f"    {sc:<25} t={best_t:.2f}  val_MCC={best_mcc:+.4f}")
    return np.array(thresholds)

def calculate_mcc(y_true, y_pred, verbose=True):
    scores = [matthews_corrcoef(y_true[:, i], y_pred[:, i])
              for i in range(y_true.shape[1])]
    if verbose:
        for sc, s in zip(SUPERCLASSES, scores):
            print(f"    {sc:<25} {s:+.4f}")
    return float(np.mean(scores))

# ── Model (best architecture from 0.7207 run) ─────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ch, ch//8), nn.ReLU(),
                                nn.Linear(ch//8, ch), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x.mean(2)).unsqueeze(2)

class TemporalAttention(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.a = nn.Linear(h, 1)
    def forward(self, x):
        return (x * torch.softmax(self.a(x), 1)).sum(1)

class MultiScaleCnnLstm(nn.Module):
    def __init__(self, n_sensors=10, n_classes=6):
        super().__init__()
        def branch(k):
            return nn.Sequential(
                nn.Conv1d(n_sensors, 64, k, padding=k//2),
                nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 128, k, padding=k//2),
                nn.BatchNorm1d(128), nn.ReLU(),
                nn.MaxPool1d(2), nn.Dropout(0.25))
        self.bs = branch(3)
        self.bm = branch(7)
        self.bl = branch(15)
        self.ca = ChannelAttention(384)
        self.cm = nn.Sequential(
            nn.Conv1d(384, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.25),
            nn.Conv1d(256, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.25))
        self.lstm = nn.LSTM(256, 128, 2, batch_first=True, dropout=0.3, bidirectional=True)
        self.ta   = TemporalAttention(256)
        self.cls  = nn.Sequential(
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_classes))

    def forward(self, x):
        x = x.permute(0,2,1)
        s,m,l = self.bs(x), self.bm(x), self.bl(x)
        t = min(s.size(2), m.size(2), l.size(2))
        x = self.ca(torch.cat([s[:,:,:t],m[:,:,:t],l[:,:,:t]], 1))
        x = self.cm(x).permute(0,2,1)
        x, _ = self.lstm(x)
        return self.cls(self.ta(x))

class SmoothBCE(nn.Module):
    def __init__(self, pw, s=0.05):
        super().__init__()
        self.pw, self.s = pw, s
    def forward(self, logits, y):
        y = y*(1-self.s) + 0.5*self.s
        return nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pw)

def train_model(X_tr, y_tr, X_tune, y_tune, epochs=80, batch_size=256):
    X_tr, y_tr = augment_minority(X_tr, y_tr, factor=4)
    print(f"    After augmentation: {len(X_tr):,}")

    model     = MultiScaleCnnLstm(len(ALL_COLS), len(SUPERCLASSES)).to(DEVICE)
    criterion = SmoothBCE(compute_pos_weights(y_tr))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    dl = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                    batch_size=batch_size, shuffle=True, num_workers=0)

    best_mcc, best_w, patience = -1, None, 0

    for epoch in range(1, epochs+1):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Evaluate on tune set every 5 epochs
        if epoch % 5 == 0:
            probs = get_probs(model, X_tune)
            preds = (probs > 0.5).astype(int)
            mcc   = calculate_mcc(y_tune, preds, verbose=False)
            if mcc > best_mcc:
                best_mcc = mcc
                best_w   = {k: v.clone() for k,v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if epoch % 20 == 0:
                print(f"    Ep {epoch:03d}/{epochs} | tune_MCC {mcc:+.4f} | "
                      f"best {best_mcc:+.4f} | pat {patience}/12")
            if patience >= 12:
                print(f"    Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_w)
    return model

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
with open(r'data\cps_data_multi_label.pkl', 'rb') as f:
    meta = pickle.load(f)
exp_dfs = {row['experiment']: prepare_experiment(row) for _, row in meta.iterrows()}
print("Done.\n")

# ── Leave-One-Out CV ──────────────────────────────────────────────────────────
test_mccs = []

for fold in range(1, 5):
    test_id   = fold
    # ✅ KEY CHANGE: train on ALL 3 non-test experiments
    train_ids = [i for i in range(1, 5) if i != test_id]

    print(f"\n{'='*60}")
    print(f"Fold {fold}/4 | Train={train_ids} Test={test_id}")

    # Concatenate all 3 training experiments
    train_df = pd.concat([exp_dfs[i] for i in train_ids], ignore_index=True)
    test_df  = exp_dfs[test_id]

    X_all, y_all, scaler = make_windows(train_df, fit_scaler=True, stride=STRIDE_TRAIN)
    X_test, y_test, _    = make_windows(test_df,  scaler=scaler,   stride=STRIDE_TEST)

    # ✅ KEY CHANGE: use last 20% of training windows for threshold tuning
    # (time-based split — no future leakage since windows are ordered by time)
    split = int(len(X_all) * 0.80)
    X_tr,   y_tr   = X_all[:split], y_all[:split]
    X_tune, y_tune = X_all[split:], y_all[split:]

    print(f"  Train:{len(X_tr):,} | Tune:{len(X_tune):,} | Test:{len(X_test):,}")

    model = train_model(X_tr, y_tr, X_tune, y_tune)

    # Tune thresholds on the held-out 20%
    tune_probs = get_probs(model, X_tune)
    thresholds = find_best_thresholds(tune_probs, y_tune)

    # Evaluate on test
    test_probs = get_probs(model, X_test)
    y_pred     = (test_probs > thresholds).astype(int)

    print(f"  Per-class MCC (test):")
    mcc = calculate_mcc(y_test, y_pred)
    test_mccs.append(mcc)
    print(f"  Fold MCC: {mcc:+.4f}")
    torch.save(model.state_dict(), f'model_fold{fold}.pt')

print(f"\n{'='*60}")
print(f"Fold MCCs : {[f'{m:+.4f}' for m in test_mccs]}")
print(f"FINAL MCC : {np.mean(test_mccs):+.4f}")