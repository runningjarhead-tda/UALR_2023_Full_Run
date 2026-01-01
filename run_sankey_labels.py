#!/usr/bin/env python
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import json
import warnings

warnings.filterwarnings("ignore")

# Config
OUT_DIR = Path("fy2023_analysis_results")
CLUSTER_FILE = OUT_DIR / "mapper_clusters_global.csv"
SEMANTIC_FILE = OUT_DIR / "PC_Semantic_Labels.json" # <--- The connection
OUTPUT_HTML = OUT_DIR / "sankey_global_pc1.html"

def main():
    print(f"[SANKEY] Loading {CLUSTER_FILE}...")
    if not CLUSTER_FILE.exists():
        print(f"Error: Could not find {CLUSTER_FILE}")
        return

    df = pd.read_csv(CLUSTER_FILE)

    # 1. Get the Semantic Name for PC1
    pc1_name = "Global Severity (PC1)" # Default fallback
    if SEMANTIC_FILE.exists():
        try:
            with open(SEMANTIC_FILE, 'r') as f:
                lookup = json.load(f)
            # Look for "PC1" or "1" in the keys
            if "PC1" in lookup:
                pc1_name = lookup["PC1"]
            elif "1" in lookup:
                pc1_name = lookup["1"]
            print(f"[SANKEY] Found semantic label: '{pc1_name}'")
        except Exception as e:
            print(f"[SANKEY] Could not read semantic labels: {e}")
    else:
        print("[SANKEY] No semantic labels found (using default). Run 'label_pcs_semantically.py' to fix.")

    # 2. Bin PC1 (Low to High) with Readable Labels
    n_bins = 6
    # Human-readable adjectives for the gradient
    adjectives = ["Lowest", "Low", "Low-Mid", "High-Mid", "High", "Highest"]
    
    bin_labels = [f"{adj} {pc1_name}" for adj in adjectives]
    
    try:
        df["PC1_Bin"] = pd.qcut(df["PCA1"], q=n_bins, labels=bin_labels)
    except Exception as e:
        print(f"Error binning PC1: {e}")
        return

    # 3. Count Flows
    flows = df.groupby(["PC1_Bin", "Cluster"]).size().reset_index(name="Count")
    
    # 4. Map to Indices for Plotly
    all_nodes = list(flows["PC1_Bin"].unique()) + list(flows["Cluster"].unique())
    node_map = {name: i for i, name in enumerate(all_nodes)}
    
    sources = [node_map[x] for x in flows["PC1_Bin"]]
    targets = [node_map[x] for x in flows["Cluster"]]
    values = flows["Count"].tolist()

    # 5. Plot
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

    fig.update_layout(title_text=f"Flow of {pc1_name} to Topological Clusters", font_size=10)
    fig.write_html(OUTPUT_HTML)
    print(f"[SANKEY] ✓ Saved diagram to {OUTPUT_HTML}")

if __name__ == "__main__":
    main()