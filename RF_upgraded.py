import pickle
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
DS_FACTOR    = 25
SEQ_LEN      = 160
SENSOR_COLS  = ['Acc.x','Acc.y','Acc.z','Gyro.x','Gyro.y','Gyro.z','Baro.x']
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

# ── Preprocessing ─────────────────────────────────────────────────────────────
def apply_superclass(df):
    df = df.copy()
    for super_name in SUPERCLASSES:
        children = [k for k, v in SUPERCLASS_MAPPING.items() if v == super_name]
        existing = [c for c in children if c in df.columns]
        df[super_name] = df[existing].max(axis=1) if existing else 0
    cols_to_drop = [k for k in SUPERCLASS_MAPPING
                    if k in df.columns and k not in SUPERCLASSES]
    df.drop(columns=cols_to_drop, inplace=True)
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

def prepare_experiment(row):
    df = row['data'].copy()
    df.drop(columns=['transportation','container','No loading'], errors='ignore', inplace=True)
    df = apply_superclass(df)
    df = downsample_df(df, DS_FACTOR)
    return df

def make_windows(df, scaler=None, fit_scaler=False):
    if fit_scaler:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df[SENSOR_COLS] = scaler.fit_transform(df[SENSOR_COLS])
    else:
        df[SENSOR_COLS] = scaler.transform(df[SENSOR_COLS])

    data_vals  = df[SENSOR_COLS].values
    label_vals = df[SUPERCLASSES].values

    X = sliding_window_view(data_vals, window_shape=SEQ_LEN, axis=0)
    # X shape: (n_windows, n_sensors, SEQ_LEN) → transpose to (n_windows, SEQ_LEN, n_sensors)
    X = X.transpose(0, 2, 1)
    y = label_vals[SEQ_LEN - 1: SEQ_LEN - 1 + len(X)]
    return X, y, scaler

# ── Improved Feature Extraction (vs their 4-feature baseline) ─────────────────
def extract_features(X):
    """X: (n_windows, SEQ_LEN, n_sensors)"""
    mean   = X.mean(axis=1)
    std    = X.std(axis=1)
    maxv   = X.max(axis=1)
    minv   = X.min(axis=1)
    rms    = np.sqrt((X**2).mean(axis=1))
    ptp    = maxv - minv
    # FFT energy per sensor
    fft_energy = np.abs(np.fft.rfft(X, axis=1)).mean(axis=1)
    return np.hstack([mean, std, maxv, minv, rms, ptp, fft_energy])
    # → 7 sensors × 7 stats = 49 features (vs their 28)

# ── MCC Calculation ───────────────────────────────────────────────────────────
def calculate_mcc(y_true, y_pred):
    scores = [matthews_corrcoef(y_true[:, i], y_pred[:, i])
              for i in range(y_true.shape[1])]
    print(f"  Per-class MCC:")
    for sc, s in zip(SUPERCLASSES, scores):
        print(f"    {sc:<25} {s:+.4f}")
    return np.mean(scores)

# ── Load all experiments ──────────────────────────────────────────────────────
print("Loading and preprocessing all experiments...")
with open(r'data\cps_data_multi_label.pkl', 'rb') as f:
    meta = pickle.load(f)

exp_dfs = {}
for _, row in meta.iterrows():
    exp_dfs[row['experiment']] = prepare_experiment(row)
print("Done.\n")

# ── Leave-One-Out Cross Validation ───────────────────────────────────────────
test_mccs = []

for fold in range(1, 5):
    test_id = fold
    val_id  = fold + 1 if fold < 4 else 1
    train_ids = [i for i in range(1, 5) if i != test_id and i != val_id]

    print(f"{'='*55}")
    print(f"Fold {fold}/4 | Train={train_ids} Val={val_id} Test={test_id}")

    train_df = pd.concat([exp_dfs[i] for i in train_ids], ignore_index=True)
    val_df   = exp_dfs[val_id].copy()
    test_df  = exp_dfs[test_id].copy()

    X_train, y_train, scaler = make_windows(train_df, fit_scaler=True)
    X_val,   y_val,   _      = make_windows(val_df,   scaler=scaler)
    X_test,  y_test,  _      = make_windows(test_df,  scaler=scaler)

    # Combine train + val for final training (matches their approach)
    X_tr = np.concatenate([X_train, X_val])
    y_tr = np.concatenate([y_train, y_val])

    print(f"  Train windows : {len(X_tr):,}")
    print(f"  Test  windows : {len(X_test):,}")

    # Feature extraction
    X_tr_feat   = extract_features(X_tr)
    X_test_feat = extract_features(X_test)

    print(f"  Feature shape : {X_tr_feat.shape[1]} features per window")

    # ── Random Forest (improved) ──────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=200,      # vs their 50
        n_jobs=-1,
        random_state=42,
        class_weight='balanced',  # handle Stationary dominance
        max_depth=None,
        min_samples_leaf=2,
    )
    model.fit(X_tr_feat, y_tr)
    y_pred = model.predict(X_test_feat)

    mcc = calculate_mcc(y_test, y_pred)
    test_mccs.append(mcc)
    print(f"  Fold MCC: {mcc:+.4f}")

print(f"\n{'='*55}")
print(f"Fold MCCs : {[f'{m:+.4f}' for m in test_mccs]}")
print(f"FINAL MCC : {np.mean(test_mccs):+.4f}")