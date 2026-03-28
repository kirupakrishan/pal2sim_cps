import pickle
import pandas as pd
import numpy as np

with open(r'data\cps_data_multi_label.pkl', 'rb') as f:
    meta = pickle.load(f)

DS_FACTOR   = 25
SEQ_LEN     = 160
SENSOR_COLS = ['Acc.x','Acc.y','Acc.z','Gyro.x','Gyro.y','Gyro.z','Baro.x']

SUPERCLASS_MAPPING = {
    "Driving(curve)":                    "Driving(curve)",
    "Driving(straight)":                 "Driving(straight)",
    "Lifting(lowering)":                 "Lifting(lowering)",
    "Lifting(raising)":                  "Lifting(raising)",
    "Wrapping":                          "Turntable wrapping",
    "Wrapping(preparation)":             "Stationary processes",
    "Docking":                           "Stationary processes",
    "Forks(entering or leaving front)":  "Stationary processes",
    "Forks(entering or leaving side)":   "Stationary processes",
    "Standing":                          "Stationary processes",
}
SUPERCLASSES = sorted(set(SUPERCLASS_MAPPING.values()))

def apply_superclass(df):
    df = df.copy()
    for super_name in SUPERCLASSES:
        children = [k for k, v in SUPERCLASS_MAPPING.items() if v == super_name]
        existing = [c for c in children if c in df.columns]
        df[super_name] = df[existing].max(axis=1) if existing else 0

    # ✅ Only drop raw labels that are NOT also a superclass name
    cols_to_drop = [k for k in SUPERCLASS_MAPPING
                    if k in df.columns and k not in SUPERCLASSES]
    df.drop(columns=cols_to_drop, inplace=True)
    return df

def downsample_df(df, factor):
    ds_rows = []
    n = len(df)
    for start in range(0, n - factor, factor):
        chunk = df.iloc[start:start + factor]
        row = {}
        for col in SENSOR_COLS:
            row[col] = chunk[col].mean()
        for col in SUPERCLASSES:
            row[col] = int(chunk[col].max())
        ds_rows.append(row)
    return pd.DataFrame(ds_rows)

# Test on experiment 1
row = meta[meta['experiment'] == 1].iloc[0]
df = row['data'].copy()
df.drop(columns=['transportation', 'container', 'No loading'], errors='ignore', inplace=True)
df = apply_superclass(df)

print("Columns after superclass mapping:", list(df.columns))
print("SUPERCLASSES present:", all(sc in df.columns for sc in SUPERCLASSES))

df_ds = downsample_df(df, DS_FACTOR)

print(f"\nOriginal   : {len(df):,} rows @ ~2027 Hz")
print(f"Downsampled: {len(df_ds):,} rows @ ~80 Hz")
print(f"Ratio      : {len(df)/len(df_ds):.1f}x")
print(f"\nSuperclass counts after downsampling:")
for sc in SUPERCLASSES:
    print(f"  {sc:<25} {int(df_ds[sc].sum()):>6,}")
print(f"\nWindow shape: ({SEQ_LEN}, {len(SENSOR_COLS)}) = {SEQ_LEN/80:.1f} sec windows ✓")