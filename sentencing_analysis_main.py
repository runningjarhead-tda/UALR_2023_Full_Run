# -*- coding: utf-8 -*-
# SENTENCING ANALYSIS - CLEAN VERSION (Based on your v4 script)
import os, sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Suppress warnings
warnings.filterwarnings("ignore")

# --- ML IMPORTS ---
# Robust GPU Check
try:
    import cudf
    from cuml.decomposition import PCA as CUMLPCA
    from cuml.preprocessing import StandardScaler as CUMLStandardScaler
    GPU_AVAILABLE = True
    print("[LOG] cuML imported successfully. GPU mode enabled.")
except ImportError:
    GPU_AVAILABLE = False
    print("[LOG] cuML not found. CPU mode enabled.")

import xgboost
import shap
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler as SklearnScaler

# ================== CONFIGURATION ==================
SEED = 42
TARGET_CHOICES = ["SENTTOT", "TOTSENTN", "SENTENCE_MONTHS"]
MAX_PCA_COMPONENTS = 20

RACE_CODE_MAP = {
    1: "White", 2: "Black", 3: "American Indian/Alaskan Native", 
    4: "Asian or Pacific Islander", 5: "Multi-racial", 7: "Other", 
    8: "Not Available", 9: "Non-US"
}

def _ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def log(msg):
    print(f"[LOG] {msg}", flush=True)

def smart_read(filepath):
    f = Path(filepath)
    if f.suffix == '.parquet': return pd.read_parquet(f)
    return pd.read_csv(f, low_memory=False)

def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

# ================== PART 1: DEMOGRAPHICS ==================
def generate_demographic_breakdown(df, target_col, out_dir):
    if 'MONRACE' not in df.columns: return
    
    # PATCH 1: Ensure target is numeric strictly for this table to prevent crash
    # We do NOT overwrite the main dataframe here, just use a temp series for stats
    df_temp = df.copy()
    df_temp[target_col] = pd.to_numeric(df_temp[target_col], errors='coerce')
    
    # Map Race
    df_temp['Race_Str'] = df_temp['MONRACE'].map(RACE_CODE_MAP).fillna("Unknown")
    
    # GroupBy (Now safe because target is numeric)
    stats = df_temp.groupby('Race_Str')[target_col].agg(['count', 'mean', 'median', 'std']).reset_index()
    stats.columns = ['Race', 'Count', 'Mean_Sentence', 'Median_Sentence', 'Std_Dev']
    
    stats.to_csv(out_dir / "demographic_breakdown.tsv", sep='\t', index=False)
    log("Saved demographic table.")

# ================== PART 2: MAIN ANALYSIS ==================
def tda_xai_full(df, out_dir, target_col):
    # PATCH 2: Fix Scope Error
    global GPU_AVAILABLE
    
    out_dir = Path(out_dir)
    log(f"Starting Analysis on {target_col}...")
    
    # Standard Clean (Your original logic)
    df_clean = df.dropna(subset=[target_col])
    
    # Ensure target is numeric for the model
    df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[target_col])
    
    # Detect Features
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    exclude = [target_col, 'USSCID', 'ID', 'FY', 'guid'] 
    features = [c for c in numeric_cols if c not in exclude]
    
    # Fallback: If 0 numeric features found, it means data loaded as text.
    # We MUST convert to run PCA.
    if len(features) == 0:
        log("Warning: No numeric features found. Attempting to auto-detect numeric columns...")
        for col in df_clean.columns:
            if col not in exclude:
                # Try convert, if mostly numbers, keep it
                tmp = pd.to_numeric(df_clean[col], errors='coerce')
                if tmp.notna().sum() > (0.5 * len(tmp)): # If >50% valid numbers
                    df_clean[col] = tmp
        # Re-select
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        features = [c for c in numeric_cols if c not in exclude]
    
    log(f"Selected {len(features)} features for PCA.")
    
    X = df_clean[features].fillna(0)
    y = df_clean[target_col]
    
    n_components = min(MAX_PCA_COMPONENTS, len(features))
    
    # --- PCA (GPU with Fallback) ---
    lens = None
    pca_model = None
    
    if GPU_AVAILABLE:
        try:
            log("Running GPU PCA...")
            import cudf # Import here to be safe
            X_gpu = cudf.DataFrame.from_pandas(X).astype('float32')
            
            scaler = CUMLStandardScaler()
            X_scaled_gpu = scaler.fit_transform(X_gpu)
            
            pca = CUMLPCA(n_components=n_components)
            lens_gpu = pca.fit_transform(X_scaled_gpu)
            
            lens = lens_gpu.to_numpy()
            pca_model = pca
            log("GPU PCA Successful.")
        except Exception as e:
            log(f"GPU failed ({e}). Switching to CPU.")
            GPU_AVAILABLE = False
            
    if not GPU_AVAILABLE:
        log("Running CPU PCA...")
        scaler = SklearnScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = SklearnPCA(n_components=n_components)
        lens = pca.fit_transform(X_scaled)
        pca_model = pca

    # PATCH 3: Save Bridge File for Mapper
    log("Saving Bridge File for Mapper...")
    proj_cols = [f'PC{i+1}' for i in range(n_components)]
    df_projections = pd.DataFrame(lens, columns=proj_cols, index=df_clean.index)
    df_projections[target_col] = y
    df_projections.to_csv(out_dir / "pca_projections_for_tda.csv")
    
    # Save Model
    try:
        joblib.dump(pca_model, out_dir / "pca_model.joblib")
        with open(out_dir / "feature_names.json", "w") as f:
            json.dump(features, f)
    except: pass

    # --- XGBoost & SHAP ---
    log("Running XGBoost & SHAP...")
    # Use safe tree method
    model = xgboost.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=500, 
        max_depth=6,
        tree_method='hist' 
    )
    model.fit(lens, y)
    
    r2 = r2_score(y, model.predict(lens))
    log(f"Model R2: {r2:.4f}")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(lens)
    
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({'Feature': proj_cols, 'SHAP_Importance': mean_shap})
    shap_df = shap_df.sort_values(by='SHAP_Importance', ascending=False)
    shap_df['SHAP_Rank'] = range(1, len(shap_df) + 1)
    
    shap_df.to_csv(out_dir / "Table_2_SHAP_Importance.csv", index=False)
    log("Analysis Complete.")

def run_all(input_path):
    out_dir = _ensure_dir("fy2023_analysis_results")
    df = smart_read(input_path)
    
    target_col = first_existing(df, TARGET_CHOICES)
    if not target_col: 
        log("Target not found.")
        return
    
    generate_demographic_breakdown(df, target_col, out_dir)
    tda_xai_full(df, out_dir, target_col)

if __name__ == "__main__":
    run_all("opafy23nid.parquet")