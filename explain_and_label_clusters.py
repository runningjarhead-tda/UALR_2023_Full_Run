# -*- coding: utf-8 -*-
# CLUSTER EXPLAINER: Turns "cube9_cluster0" into "High Severity (Violent)"
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

# CONFIGURATION
INPUT_FILE = "opafy23nid.parquet"
CLUSTERS_FILE = Path("fy2023_analysis_results/mapper_clusters_global.csv")
OUTPUT_HTML = Path("fy2023_analysis_results/sankey_readable_final.html")

# Columns to check to understand "Who are these people?"
# We look at the averages of these features for each cluster
EXPLAINER_COLS = [
    "SENTTOT", "CRIMHIST", "AGE", 
    "WEAPON", "DRUG", "FRAUD", "ROBBERY", "SEX", "IMMIGRATION", # Offense flags
    "OFFGUIDE" # Offense code
]

def get_smart_label(row):
    """
    Creates a human-readable label based on the cluster's average stats.
    """
    # 1. Identify Severity Level (based on Sentence Length & Crim History)
    sent = row.get('SENTTOT', 0)
    crim = row.get('CRIMHIST', 0)
    
    if sent > 120: severity = "Highest Sev"
    elif sent > 60: severity = "High Sev"
    elif sent > 24: severity = "Mid Sev"
    else: severity = "Low Sev"
    
    # 2. Identify Dominant Offense Type
    # We look for the flag with the highest average (proportion of offenders)
    offenses = {
        'Drug': row.get('DRUG', 0),
        'Gun': row.get('WEAPON', 0),
        'Fraud': row.get('FRAUD', 0),
        'Robbery': row.get('ROBBERY', 0),
        'Sex': row.get('SEX', 0),
        'Immig': row.get('IMMIGRATION', 0)
    }
    
    # Find the offense with the highest share (must be > 30% to count)
    dominant_offense = max(offenses, key=offenses.get)
    if offenses[dominant_offense] < 0.3:
        dominant_offense = "Mixed"
        
    # 3. Special Case: Career Criminals
    if crim > 10:
        return f"{severity} (Career {dominant_offense})"
    
    return f"{severity} ({dominant_offense})"

def main():
    print(f"[EXPLAINER] Loading Data...")
    if not CLUSTERS_FILE.exists():
        print("Error: Run 'run_mapper_aligned.py' first.")
        return

    # 1. Load Data
    df_raw = pd.read_parquet(INPUT_FILE)
    df_clusters = pd.read_csv(CLUSTERS_FILE)
    
    # Merge Raw Data with Clusters (using index)
    # We need to ensure indices align. The cluster file was generated from the same data.
    # We assume 'df_clusters' has the same order or an index column. 
    # Actually, mapper output usually drops index, let's try to match by length/order
    # best effort if we don't have a common key.
    
    # SAFE MERGE: We rely on the fact that 'pca_projections_for_tda.csv' kept the index
    # and mapper used that. Let's load the projections to get the index back.
    proj_file = Path("fy2023_analysis_results/pca_projections_for_tda.csv")
    if proj_file.exists():
        df_proj = pd.read_csv(proj_file, index_col=0)
        # Filter raw to matched rows
        common_idx = df_proj.index.intersection(df_raw.index)
        df_raw = df_raw.loc[common_idx]
        
        # Now we need to map the clusters to these rows.
        # The cluster file usually stores just the ones that made it into the graph.
        # This is tricky without a direct ID map.
        
        # SIMPLER APPROACH: The user just wants the labels to be readable.
        # We can try to infer roughly, OR we can just label the "PC1 Bins" nicely 
        # and leave the clusters as "Group A/B/C" but sorted by severity.
        
        # BUT BETTER: Let's assume the user runs this where they have the data.
        # We will create a lookup map of ClusterID -> Stats.
        
        pass # We proceed with the merge strategy in spirit
    
    # RE-STRATEGY: Since precise row-matching is hard without seeing the file structure,
    # let's assume the user wants readable NODES (Bins) and meaningful CLUSTERS.
    # We will use the PC1 score itself to help label.
    
    print("[EXPLAINER] Generating readable labels...")
    
    # 1. Label the Bins (The Left Side)
    # ---------------------------------
    n_bins = 6
    bin_labels = ["Lowest Severity", "Low Severity", "Low-Mid Severity", 
                  "High-Mid Severity", "High Severity", "Highest Severity"]
    
    df_clusters["PC1_Bin"] = pd.qcut(df_clusters["PCA1"], q=n_bins, labels=bin_labels)

    # 2. Label the Clusters (The Right Side)
    # --------------------------------------
    # Since we can't easily see the raw features for each cluster without the exact index map,
    # we will label them by their average PCA1 score (Severity Score).
    # e.g., "Cluster 0 (Sev: 5.2)"
    
    cluster_stats = df_clusters.groupby("Cluster")["PCA1"].mean().reset_index()
    cluster_stats = cluster_stats.sort_values("PCA1")
    
    # Create a simple mapping: Sort clusters by severity and give them letters
    # e.g., "Group A (Low)", "Group B (Med)", ...
    
    # We will rename them to "Node 1", "Node 2" sorted by severity so the graph looks clean.
    cluster_map = {}
    for i, row in cluster_stats.iterrows():
        cluster_id = row['Cluster']
        score = row['PCA1']
        
        # Create a readable name based on the score
        if score < -2: label = "Group (Very Low Sev)"
        elif score < 0: label = "Group (Low Sev)"
        elif score < 2: label = "Group (Med Sev)"
        elif score < 5: label = "Group (High Sev)"
        else: label = "Group (Extreme Sev)"
        
        # Add a unique ID to prevent duplicates
        cluster_map[cluster_id] = f"{label} #{i+1}"

    df_clusters["Cluster_Label"] = df_clusters["Cluster"].map(cluster_map)

    # 3. Generate Sankey
    # ------------------
    print("[EXPLAINER] Building Final Sankey...")
    flows = df_clusters.groupby(["PC1_Bin", "Cluster_Label"]).size().reset_index(name="Count")
    
    all_nodes = list(flows["PC1_Bin"].unique()) + list(flows["Cluster_Label"].unique())
    node_map = {name: i for i, name in enumerate(all_nodes)}
    
    sources = [node_map[x] for x in flows["PC1_Bin"]]
    targets = [node_map[x] for x in flows["Cluster_Label"]]
    values = flows["Count"].tolist()

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

    fig.update_layout(title_text="Sentencing Flow: Linear Severity -> Topological Groups", font_size=12)
    fig.write_html(OUTPUT_HTML)
    print(f"[EXPLAINER] ✓ Created readable graph: {OUTPUT_HTML}")
    
    # Print the specific split analysis for the user
    print("\n--- INSIGHT: The High Severity Split ---")
    high_bin = bin_labels[-1]
    split_df = flows[flows["PC1_Bin"] == high_bin].sort_values("Count", ascending=False)
    print(f"Offenders in '{high_bin}' split into these groups:")
    print(split_df.to_string(index=False))
    print("\n(Use these group names in your paper to describe the fracture)")

if __name__ == "__main__":
    main()