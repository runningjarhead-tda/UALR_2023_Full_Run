#!/usr/bin/env python
import pandas as pd
import numpy as np
import kmapper as km
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings

# Suppress minor warnings
warnings.filterwarnings("ignore")

# ================== CONFIGURATION ==================
DATA_FILE = "opafy23nid.parquet"           # Your raw data file
INPUT_DIR = Path("fy2023_analysis_results") 
PROJECTIONS_FILE = INPUT_DIR / "pca_projections_for_tda.csv" # OUTPUT FROM MAIN SCRIPT

# Output settings
OUT_DIR = INPUT_DIR
MAPPER_HTML = OUT_DIR / "mapper_fy2023_global.html"
CLUSTER_CSV = OUT_DIR / "mapper_clusters_global.csv" # INPUT FOR SANKEY

# Features to color the nodes by (Content)
FEATURES = [
    "SENTTOT", "SENTMON", "CRIMHIST", 
    "AGE", "WEAPON", "DRUG", "VICTIM", "NUMCOUNTS"
]

def log(msg): print(f"[MAPPER] {msg}", flush=True)

def main():
    # 1. Load Global Projections (The Lens)
    if not PROJECTIONS_FILE.exists():
        log(f"CRITICAL: Could not find {PROJECTIONS_FILE}")
        log("Run 'sentencing_analysis_main.py' first!")
        return

    log("Loading Global PCA Lens...")
    df_proj = pd.read_csv(PROJECTIONS_FILE, index_col=0)
    
    # 2. Load Raw Data (The Content)
    log("Loading Raw Data...")
    # Using pandas for compatibility.
    df_raw = pd.read_parquet(DATA_FILE)
    
    # 3. Align Data (Intersection of indices)
    common_idx = df_proj.index.intersection(df_raw.index)
    log(f"Aligned Data: {len(common_idx)} rows match between Global Model and Raw Data.")
    
    df_lens = df_proj.loc[common_idx]
    df_content = df_raw.loc[common_idx]

    # 4. Setup Mapper Inputs
    # Lens: Global PC1 (Calculated in Main Analysis)
    lens = df_lens[['PC1']].values 
    
    # Content: Selected Features (For clustering distance)
    valid_feats = [c for c in FEATURES if c in df_content.columns]
    
    # === FIX: FORCE NUMERIC CONVERSION ===
    # We create a clean subset and force-convert strings to numbers
    X_content = df_content[valid_feats].copy()
    for col in X_content.columns:
        X_content[col] = pd.to_numeric(X_content[col], errors='coerce')
    
    # NOW it is safe to fill NaNs with 0
    X_content = X_content.fillna(0)
    X_scaled = StandardScaler().fit_transform(X_content)

    # 5. Run Mapper
    log("Running KeplerMapper...")
    mapper = km.KeplerMapper(verbose=1)
    
    # Parameters: Slices PC1 into 20 overlapping intervals
    graph = mapper.map(
        lens,
        X_scaled,
        cover=km.Cover(n_cubes=20, perc_overlap=0.3),
        clusterer=DBSCAN(eps=0.5, min_samples=10)
    )

    # 6. Save Graph
    # For tooltips, we want the original (readable) values, so we use df_content
    # assuming we might want to see the original strings if they existed, 
    # but X_content is safer for plotting.
    mapper.visualize(
        graph,
        path_html=str(MAPPER_HTML),
        title="FY2023 TDA: Global PC1 Lens",
        custom_tooltips=X_content.astype(str).values
    )
    log(f"✓ Saved Graph to {MAPPER_HTML}")

    # 7. Extract Clusters for Sankey
    log("Extracting clusters for Sankey...")
    rows = []
    
    # Map every data point in a node to that node's ID
    for node_id, member_indices in graph['nodes'].items():
        for idx in member_indices:
            # idx is the position in the lens array
            rows.append({
                'PCA1': lens[idx][0], # The Global PC1 Value
                'Cluster': node_id
            })
            
    df_clusters = pd.DataFrame(rows)
    # Simplify: If a point is in multiple clusters, keep the first one
    df_clusters = df_clusters.drop_duplicates(subset=['PCA1'])
    
    df_clusters.to_csv(CLUSTER_CSV, index=False)
    log(f"✓ Saved Cluster Data to {CLUSTER_CSV}")

if __name__ == "__main__":
    main()