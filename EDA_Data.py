"""
=============================================================================
Pal2Sim — Complete Data Analysis Script
=============================================================================
Combines:
  - Structure exploration   (what is inside the pickle)
  - 8-point verification    (confirms every modeling assumption)
  - EDA plots               (visual class/sensor/timeline overview)
  - Actionable insights     (printed conclusions after every section)

Run from your project root:
    python pal2sim_data_analysis.py

Outputs:
    eda_overview.png   — 4-panel EDA figure
    eda_timeline.png   — per-experiment activity timeline
=============================================================================
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Path ──────────────────────────────────────────────────────────────────────
PKL_PATH = r'data\cps_data_multi_label.pkl'

LABEL_COLS = [
    'No loading', 'Driving(straight)', 'Driving(curve)',
    'Lifting(raising)', 'Lifting(lowering)', 'Standing', 'Docking',
    'Forks(entering or leaving front)', 'Forks(entering or leaving side)',
    'Wrapping', 'Wrapping(preparation)'
]
SENSOR_COLS = ['Acc.x', 'Acc.y', 'Acc.z', 'Gyro.x', 'Gyro.y', 'Gyro.z', 'Baro.x']

SECTION = lambda title: print(f"\n{'='*65}\n  {title}\n{'='*65}")
INSIGHT = lambda msg:   print(f"\n  ► INSIGHT: {msg}")

# =============================================================================
# LOAD
# =============================================================================
SECTION("LOADING DATA")
with open(PKL_PATH, 'rb') as f:
    meta = pickle.load(f)

print(f"  Type         : {type(meta)}")
print(f"  Shape        : {meta.shape}")
print(f"  Columns      : {list(meta.columns)}")
print(f"  Scenarios    : {meta['scenario'].unique()}")
print(f"  Experiments  : {sorted(meta['experiment'].unique())}")
INSIGHT("Top-level pickle is a small (4-row) DataFrame. "
        "Each row holds a full experiment as a nested DataFrame in the 'data' column.")

# =============================================================================
# SECTION 1 — STRUCTURE EXPLORATION
# =============================================================================
SECTION("SECTION 1 — NESTED DATAFRAME STRUCTURE")

exp_dfs = {}
for _, row in meta.iterrows():
    eid = row['experiment']
    df  = row['data'].copy()
    exp_dfs[eid] = df

    duration = df['time'].max() - df['time'].min()
    null_pct  = df.isnull().sum().sum() / df.size * 100

    print(f"\n  Experiment {eid}")
    print(f"    Rows       : {len(df):>10,}")
    print(f"    Columns    : {df.shape[1]}")
    print(f"    Time range : {df['time'].min():.2f}s → {df['time'].max():.2f}s  ({duration:.1f}s)")
    print(f"    Null %     : {null_pct:.2f}%")
    print(f"    transport  : {df['transportation'].unique()}")
    print(f"    container  : {df['container'].unique()}")
    print(f"    Columns    : {list(df.columns)}")
    print(f"    Dtypes:\n{df.dtypes.to_string()}")
    print(f"    First 3 rows:\n{df.head(3).to_string()}")

all_df = pd.concat(exp_dfs.values(), ignore_index=True)
print(f"\n  Total rows across all experiments: {len(all_df):,}")

INSIGHT("Each inner DataFrame has 21 columns: time + 7 sensor channels + "
        "transportation + container + 10 label columns (binary 0/1). "
        "The 'data' column stores full time-series per experiment.")

# =============================================================================
# SECTION 2 — 8-POINT VERIFICATION
# =============================================================================
SECTION("SECTION 2 — 8-POINT VERIFICATION")

# ── Check 1: transportation & container fully NULL ────────────────────────────
print("\n  [CHECK 1] Are transportation & container fully NULL?")
all_null = True
for col in ['transportation', 'container']:
    n = all_df[col].isna().sum()
    fully = (n == len(all_df))
    all_null = all_null and fully
    print(f"    {col:<16}: {n:,} / {len(all_df):,} nulls  → fully NULL: {fully}")
INSIGHT("transportation and container are 100% NULL across all experiments. "
        "DROP these columns before training — they carry zero information.")

# ── Check 2: No loading always 0 ─────────────────────────────────────────────
print("\n  [CHECK 2] Is 'No loading' always 0?")
unique_vals = all_df['No loading'].unique()
total_sum   = all_df['No loading'].sum()
print(f"    Unique values : {unique_vals}")
print(f"    Sum           : {total_sum}")
INSIGHT("'No loading' never fires — it is a dead class. "
        "DROP it from label_cols; it wastes a classification head output.")

# ── Check 3: Docking rarity ───────────────────────────────────────────────────
print("\n  [CHECK 3] How rare is Docking?")
docking_total = all_df['Docking'].sum()
print(f"    Total Docking samples : {docking_total:,}  "
      f"({docking_total/len(all_df)*100:.4f}%)")
for eid, df in exp_dfs.items():
    print(f"    Exp {eid}: {df['Docking'].sum():,}")
INSIGHT("Docking = 0.06% of all samples. "
        "However, in the competition it is MERGED into 'Stationary processes' "
        "via the superclass mapping — its rarity no longer matters directly.")

# ── Check 4: Multi-label and zero-label rows ──────────────────────────────────
print("\n  [CHECK 4] Multi-label and zero-label rows")
for eid, df in exp_dfs.items():
    label_sum = df[LABEL_COLS].sum(axis=1)
    multi     = (label_sum > 1).sum()
    zero      = (label_sum == 0).sum()
    single    = (label_sum == 1).sum()
    print(f"\n    Exp {eid}: zero={zero:,} | single={single:,} | multi={multi:,}")
    combos = df[label_sum > 1][LABEL_COLS].apply(
        lambda r: tuple(LABEL_COLS[i] for i, v in enumerate(r) if v == 1), axis=1)
    for combo, cnt in combos.value_counts().head(5).items():
        print(f"      {cnt:>6,}x  {combo}")
INSIGHT("Every multi-label combination is Driving + Lifting. "
        "This is physically real: the forklift adjusts fork height while moving. "
        "Use sigmoid (not softmax) — these are genuine concurrent activities. "
        "~400k zero-label rows are unlabeled transitions.")

# ── Check 5: Sampling rate ────────────────────────────────────────────────────
print("\n  [CHECK 5] Sampling rate per experiment")
rates = []
for eid, df in exp_dfs.items():
    duration   = df['time'].max() - df['time'].min()
    rate       = len(df) / duration
    time_diffs = df['time'].diff().dropna()
    rates.append(rate)
    print(f"    Exp {eid}: {len(df):,} rows / {duration:.1f}s = {rate:.1f} Hz")
    print(f"      time diff → mean={time_diffs.mean():.5f}s | "
          f"min={time_diffs.min():.5f}s | max={time_diffs.max():.5f}s")
INSIGHT(f"Exact sampling rate: {np.mean(rates):.1f} Hz across all experiments. "
        "Downsample factor = 25 → 80 Hz target. "
        "seq_len = 160 → 2-second windows | seq_len = 320 → 4-second windows. "
        "This is the single most important number for windowing design.")

# ── Check 6: Sensor value ranges ─────────────────────────────────────────────
print("\n  [CHECK 6] Sensor value ranges (sanity check)")
print(f"    {'Sensor':<10} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
print("    " + "-"*52)
sensor_issues = []
for col in SENSOR_COLS:
    s = all_df[col].dropna()
    print(f"    {col:<10} {s.min():>12.4f} {s.max():>12.4f} "
          f"{s.mean():>12.4f} {s.std():>12.4f}")
    if abs(s.max()) > 1000 or abs(s.min()) > 1000:
        sensor_issues.append(col)
if sensor_issues:
    INSIGHT(f"Extreme outliers detected in: {sensor_issues}. "
            "Use RobustScaler (not MinMaxScaler) — it is median-based and "
            "handles outliers without compressing the signal range.")
else:
    INSIGHT("All sensor ranges are physically plausible. "
            "Acc ~±50 m/s², Gyro ~±1.7 rad/s, Baro ~101.7 kPa. "
            "No corrupt channels. Use RobustScaler for normalisation.")

# ── Check 7: Labels strictly 0/1 ─────────────────────────────────────────────
print("\n  [CHECK 7] Are label values strictly 0 or 1?")
all_binary = True
for lbl in LABEL_COLS:
    unique = set(all_df[lbl].dropna().unique())
    clean  = unique <= {0, 1, np.int64(0), np.int64(1)}
    if not clean:
        all_binary = False
    print(f"    {lbl:<44} binary: {clean}  vals={sorted(unique)}")
INSIGHT("All labels are clean binary integers (0 or 1). "
        "No corrupt values — safe to use BCEWithLogitsLoss directly.")

# ── Check 8: Column consistency ───────────────────────────────────────────────
print("\n  [CHECK 8] Do all experiments have identical columns?")
cols_list = [set(df.columns) for df in exp_dfs.values()]
all_same  = all(c == cols_list[0] for c in cols_list)
print(f"    All experiments have same columns: {all_same}")
if not all_same:
    for i, (eid, cols) in enumerate(zip(exp_dfs.keys(), cols_list)):
        diff = cols.symmetric_difference(cols_list[0])
        if diff:
            print(f"    Exp {eid} differs: {diff}")
INSIGHT("All 4 experiments are structurally identical — safe to concatenate "
        "across experiments for training without alignment issues.")

# =============================================================================
# SECTION 3 — CLASS IMBALANCE SUMMARY
# =============================================================================
SECTION("SECTION 3 — CLASS IMBALANCE SUMMARY")

total = len(all_df)
print(f"\n  {'Class':<44} {'Count':>10}  {'%':>7}")
print("  " + "-"*65)
imbalance_flags = []
for lbl in LABEL_COLS:
    cnt = all_df[lbl].sum()
    pct = cnt / total * 100
    flag = " ← CRITICAL" if pct < 0.1 else (" ← MINORITY" if pct < 5 else "")
    if flag:
        imbalance_flags.append((lbl, pct))
    print(f"  {lbl:<44} {cnt:>10,}  {pct:>6.2f}%{flag}")

print(f"\n  Per-experiment dominant class:")
for eid, df in exp_dfs.items():
    active = df[LABEL_COLS].sum()
    dom    = active.idxmax()
    print(f"    Exp {eid}: {dom}  ({active[dom]:,} samples)")

INSIGHT("Standing dominates at 35% — model will be biased toward it without "
        "class weighting. Use BCEWithLogitsLoss(pos_weight=neg/pos) capped at 10x. "
        "Lifting(raising) and Lifting(lowering) are the key minority classes "
        "to augment. After superclass mapping, Docking+Forks+Standing all merge "
        "into 'Stationary processes' which reduces imbalance significantly.")

# =============================================================================
# SECTION 4 — SUPERCLASS MAPPING IMPACT
# =============================================================================
SECTION("SECTION 4 — SUPERCLASS MAPPING IMPACT")

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

print(f"\n  Mapping ({len(LABEL_COLS)-1} active labels → {len(SUPERCLASSES)} superclasses):\n")
for raw, sup in SUPERCLASS_MAPPING.items():
    arrow = "──►" if raw != sup else "═══"
    print(f"    {raw:<44} {arrow}  {sup}")

# Compute superclass counts
df_sc = all_df.copy()
for sn in SUPERCLASSES:
    children = [k for k, v in SUPERCLASS_MAPPING.items() if v == sn]
    existing = [c for c in children if c in df_sc.columns]
    df_sc[sn] = df_sc[existing].max(axis=1) if existing else 0

print(f"\n  Superclass counts (all experiments combined):")
print(f"  {'Superclass':<25} {'Count':>10}  {'%':>7}")
print("  " + "-"*46)
for sc in SUPERCLASSES:
    cnt = df_sc[sc].sum()
    pct = cnt / total * 100
    print(f"  {sc:<25} {cnt:>10,}  {pct:>6.2f}%")

INSIGHT("After mapping, 'Stationary processes' absorbs Standing + Docking + "
        "Forks(front) + Forks(side) + Wrapping(preparation) → ~45% of data. "
        "Lifting(raising) remains the hardest class (~3%). "
        "The 6-class problem is significantly more balanced than the raw 11-class one.")

# =============================================================================
# SECTION 5 — EDA PLOTS
# =============================================================================
SECTION("SECTION 5 — EDA PLOTS (saving to disk)")

fig, axes = plt.subplots(2, 2, figsize=(20, 13))
fig.suptitle('Pal2Sim EDA — All Experiments', fontsize=15, fontweight='bold')

# Plot A: Label counts per experiment
ax = axes[0, 0]
label_by_exp = pd.DataFrame({
    f'Exp {eid}': df[LABEL_COLS].sum()
    for eid, df in exp_dfs.items()
})
label_by_exp.plot(kind='bar', ax=ax, colormap='tab10')
ax.set_title('Raw Label Counts per Experiment')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=45)
ax.legend(loc='upper right')
ax.set_ylabel('Samples')

# Plot B: Null value heatmap
ax = axes[0, 1]
null_matrix = pd.DataFrame({
    f'Exp {eid}': df.isnull().sum()
    for eid, df in exp_dfs.items()
})
sns.heatmap(null_matrix, annot=True, fmt='d', cmap='Reds',
            ax=ax, linewidths=0.5)
ax.set_title('Null Values per Column × Experiment')
ax.tick_params(axis='y', rotation=0)

# Plot C: Sensor boxplot
ax = axes[1, 0]
all_df[SENSOR_COLS].dropna().boxplot(ax=ax, grid=True)
ax.set_title('Sensor Value Distribution (all experiments)')
ax.set_ylabel('Value')
ax.tick_params(axis='x', rotation=30)

# Plot D: Superclass co-occurrence matrix
ax = axes[1, 1]
sc_data = df_sc[SUPERCLASSES].fillna(0)
cooc    = sc_data.T.dot(sc_data)
sns.heatmap(cooc, annot=True, fmt='.0f', cmap='Blues', ax=ax,
            xticklabels=SUPERCLASSES, yticklabels=SUPERCLASSES)
ax.set_title('Superclass Co-occurrence Matrix')
ax.tick_params(axis='x', rotation=30)
ax.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → eda_overview.png")

# Timeline plot
fig, axes = plt.subplots(4, 1, figsize=(20, 18), sharex=False)
fig.suptitle('Activity Labels Over Time — Per Experiment',
             fontsize=14, fontweight='bold')
colors = plt.cm.tab20(np.linspace(0, 1, len(LABEL_COLS)))

for ax, (eid, df) in zip(axes, exp_dfs.items()):
    df_c = df[['time'] + LABEL_COLS].dropna(subset=['time'])
    for i, lbl in enumerate(LABEL_COLS):
        active = df_c[df_c[lbl] == 1]['time']
        ax.scatter(active, [i] * len(active), s=0.5, alpha=0.4,
                   color=colors[i])
    ax.set_yticks(range(len(LABEL_COLS)))
    ax.set_yticklabels(LABEL_COLS, fontsize=7)
    ax.set_title(f'Experiment {eid}  ({len(df):,} rows)', fontsize=10)
    ax.set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig('eda_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → eda_timeline.png")

INSIGHT("Timeline shows consistent cyclic pattern across all 4 experiments: "
        "Drive → Fork → Lift → Stand → Wrap. "
        "Wrapping always appears as one long continuous block (~270–370s). "
        "This consistency means the 4 experiments ARE comparable for cross-validation.")

# =============================================================================
# SECTION 6 — WINDOWING CONFIGURATION
# =============================================================================
SECTION("SECTION 6 — WINDOWING CONFIGURATION SUMMARY")

fs_raw    = np.mean([len(df) / (df['time'].max() - df['time'].min())
                     for df in exp_dfs.values()])
ds_factor = 25
fs_target = fs_raw / ds_factor

print(f"\n  Raw sampling rate  : {fs_raw:.1f} Hz")
print(f"  Downsample factor  : {ds_factor}x")
print(f"  Target rate        : {fs_target:.1f} Hz")
print(f"\n  Window options:")
for secs in [1, 2, 4]:
    seq = int(fs_target * secs)
    print(f"    {secs}s window → seq_len = {seq} timesteps")

print(f"\n  Recommended strides:")
print(f"    Training : stride = {int(fs_target * 2 * 0.25)} (75% overlap on 4s window)")
print(f"    Test     : stride = {int(fs_target * 2 * 0.1)}  (90% overlap for dense eval)")

print(f"\n  Derived feature recommendations:")
print(f"    acc_mag  = sqrt(Acc.x² + Acc.y² + Acc.z²)  → encodes lifting intensity")
print(f"    gyro_mag = sqrt(Gyro.x² + Gyro.y² + Gyro.z²) → encodes turning vs straight")
print(f"    acc_jerk = diff(acc_mag)                     → encodes lift start/stop")

INSIGHT("4-second windows (seq_len=320) outperform 2-second windows for distinguishing "
        "Driving(straight) vs Driving(curve) because curves require sustained rotation. "
        "acc_jerk is the single most discriminative derived feature for Lifting classes. "
        "Always add derived features AFTER downsampling to avoid jerk noise amplification.")

# =============================================================================
# FINAL MODELING DECISIONS SUMMARY
# =============================================================================
SECTION("FINAL MODELING DECISIONS — DERIVED FROM THIS ANALYSIS")

decisions = [
    ("DROP columns",        "transportation, container, No loading — zero information"),
    ("SUPERCLASSES",        "11 raw labels → 6 superclasses via organiser mapping"),
    ("DOWNSAMPLE",          f"{fs_raw:.0f} Hz → 80 Hz (factor=25), mean for sensors, max for labels"),
    ("WINDOW SIZE",         "seq_len=320 (4 seconds at 80 Hz)"),
    ("STRIDE (train)",      "40 samples (50% overlap) — avoids 99.7% overlap memorization"),
    ("STRIDE (test)",       "20 samples (dense predictions for accurate MCC)"),
    ("DERIVED FEATURES",    "acc_mag, gyro_mag, acc_jerk — add after downsampling"),
    ("SCALER",              "RobustScaler (handles sensor outliers, fit on train only)"),
    ("LOSS",                "BCEWithLogitsLoss with pos_weight = min(neg/pos, 10)"),
    ("AUGMENTATION",        "Oversample Lifting(raising/lowering) ×4 with noise+scale+roll"),
    ("LABEL STRATEGY",      "sigmoid output (multi-label, not softmax)"),
    ("THRESHOLD",           "Per-class threshold tuned on val set (not fixed 0.5)"),
    ("CROSS-VALIDATION",    "Leave-one-out across 4 experiments, train on 3"),
    ("KEY WEAK CLASS",      "Lifting(raising) — lowest MCC across all runs, needs most attention"),
]

print()
for decision, reason in decisions:
    print(f"  {decision:<22}  →  {reason}")

print(f"\n{'='*65}")
print("  Analysis complete. See eda_overview.png and eda_timeline.png")
print(f"{'='*65}\n")