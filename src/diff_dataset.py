
import pandas as pd

def normalize(df):
    df.columns = df.columns.str.strip().str.lower()
    df = df.sort_index(axis=1).reset_index(drop=True)
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip().replace({'nan': pd.NA, '': pd.NA})
    return df

df_old = pd.read_excel("../res/SSBU25_dataset_adam.xlsx")
df_new = pd.read_excel("../res/SSBU25_dataset_samuel.xlsx")

df_old = normalize(df_old)
df_new = normalize(df_new)

common_cols = df_old.columns.intersection(df_new.columns)
df_old = df_old[common_cols]
df_new = df_new[common_cols]

df_old = df_old.sort_index().reset_index(drop=True)
df_new = df_new.sort_index().reset_index(drop=True)

if df_old.shape != df_new.shape:
    print(f"⚠️ DataFrames have different shapes: {df_old.shape} vs {df_new.shape}")
else:
    print(f"✅ DataFrames have the same shape: {df_old.shape}")

print("\n📊 Column Type Differences:")
dtype_diff = (df_old.dtypes != df_new.dtypes)
if dtype_diff.any():
    for col in df_old.columns[dtype_diff]:
        print(f"- {col}: old={df_old[col].dtype}, new={df_new[col].dtype}")
else:
    print("✅ No dtype differences")

print("\n🔍 Cell Value Differences:")
diff = df_old != df_new
diff_cells = diff.sum().sum()
print(f"Total differing cells: {diff_cells}")
if diff_cells > 0:
    diffs = []
    for col in df_old.columns:
        mismatches = df_old[col] != df_new[col]
        for idx in df_old[mismatches].index:
            diffs.append({
                "Row": idx,
                "Column": col,
                "Old": df_old.at[idx, col],
                "New": df_new.at[idx, col]
            })
    diff_df = pd.DataFrame(diffs)
    pd.set_option("display.max_rows", 100)
    print(diff_df)

print("\n🕳️ Missing Value Differences:")
nan_diff = df_old.isna() != df_new.isna()
nan_diff_cells = nan_diff.sum().sum()
print(f"Total mismatched NaNs: {nan_diff_cells}")
if nan_diff_cells > 0:
    nan_diffs = []
    for col in df_old.columns:
        mismatches = df_old[col].isna() != df_new[col].isna()
        for idx in df_old[mismatches].index:
            nan_diffs.append({
                "Row": idx,
                "Column": col,
                "Old is NaN": pd.isna(df_old.at[idx, col]),
                "New is NaN": pd.isna(df_new.at[idx, col])
            })
    nan_diff_df = pd.DataFrame(nan_diffs)
    pd.set_option("display.max_rows", 100)
    print(nan_diff_df)
