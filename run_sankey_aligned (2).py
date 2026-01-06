#!/usr/bin/env python
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import os

# Config
OUT_DIR = Path("fy2023_analysis_results")
CLUSTER_FILE = OUT_DIR / "mapper_clusters_global.csv" # Output from Mapper Script
OUTPUT_HTML = OUT_DIR / "sankey_global_pc1.html"

def main():
    print(f"[SANKEY] Loading {CLUSTER_FILE}...")
    if not CLUSTER_FILE.exists():
        print(f"Error: Could not find {CLUSTER_FILE}")
        print("Make sure you ran 'run_mapper_aligned.py' successfully.")
        return

    df = pd.read_csv(CLUSTER_FILE)
    print(f"[SANKEY] Loaded {len(df)} rows.")

    # 1. Bin PC1 (Low to High)
    # This creates 6 groups: "Lowest PC1 scores" -> "Highest PC1 scores"
    n_bins = 6
    try:
        df["PC1_Bin"] = pd.qcut(df["PCA1"], q=n_bins, labels=[f"PC1 Bin {i+1}" for i in range(n_bins)])
    except Exception as e:
        print(f"Error binning PC1: {e}")
        return

    # 2. Count Flows
    # How many cases flow from "Bin 1" -> "Cluster X"?
    flows = df.groupby(["PC1_Bin", "Cluster"]).size().reset_index(name="Count")
    
    # 3. Map to Indices for Plotly
    # Create a unique list of all labels (Bins + Cluster Names)
    all_nodes = list(flows["PC1_Bin"].unique()) + list(flows["Cluster"].unique())
    node_map = {name: i for i, name in enumerate(all_nodes)}
    
    sources = [node_map[x] for x in flows["PC1_Bin"]]
    targets = [node_map[x] for x in flows["Cluster"]]
    values = flows["Count"].tolist()

    # 4. Plot
    print("[SANKEY] Building diagram...")
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5),
            label=all_nodes,
            color="blue"
        ),
        link=dict(
            source=sources, target=targets, value=values
        )
    )])

    fig.update_layout(title_text="Global PC1 Flow to Topological Clusters", font_size=10)
    fig.write_html(OUTPUT_HTML)
    print(f"[SANKEY] ✓ Saved diagram to {OUTPUT_HTML}")

if __name__ == "__main__":
    main()
