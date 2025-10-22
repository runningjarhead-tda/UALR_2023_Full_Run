# --- Code Snippet to Find First Demographic PC ---
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration ---
output_folder = "fy2023_analysis_results"
shap_importance_file = "Table_2_SHAP_Importance.csv" # Make sure this filename is correct
num_features_to_show = 15 # Show more features to increase chance of finding demographics
search_terms = ['RACE', 'SEX', 'GENDER', 'HISPORIG'] # Terms to look for in feature names

# --- Helper Functions ---
def log(s): print(f"[INFO] {s}", flush=True)

def find_demographic_features(features_list, terms):
    """Checks if any demographic terms are in the top features."""
    found_features = []
    for feature in features_list:
        # Normalize feature name for robust checking
        norm_feature = feature.upper().replace('_', '')
        for term in terms:
            if term in norm_feature:
                found_features.append(feature)
                break # Move to next feature once a term is found
    return found_features

# --- Load Models and SHAP Ranking ---
out_dir = Path(output_folder)
try:
    pca = joblib.load(out_dir / "pca_model.joblib")
    with open(out_dir / "feature_names.json", 'r') as f:
        feature_names = json.load(f)
    shap_importance = pd.read_csv(out_dir / shap_importance_file)
    # Assuming first column is PC name (e.g., from index or unnamed column)
    shap_importance.rename(columns={shap_importance.columns[0]: 'PC_Name'}, inplace=True)
    log("Successfully loaded PCA model, features, and SHAP ranking.")
except FileNotFoundError as e:
    log(f"ERROR: Could not find required file: {e}")
    log("Please ensure the LATEST 'master' script ran successfully and created:")
    log(f"  - {out_dir / 'pca_model.joblib'}")
    log(f"  - {out_dir / 'feature_names.json'}")
    log(f"  - {out_dir / shap_importance_file}")
    raise
except Exception as e:
    log(f"An error occurred loading files: {e}")
    raise

# --- Iterate Through PCs Ranked 6 to 50 ---
log(f"Analyzing PCs ranked 6th to 50th for demographic features ({search_terms})...")

# Get PC names in order of SHAP importance (skip top 5)
ranked_pcs = shap_importance['PC_Name'].tolist()[5:50]

for rank, pc_name in enumerate(ranked_pcs, start=6):
    try:
        pc_index = int(pc_name.split('_')[-1]) - 1
        component = pca.components_[pc_index]
        pc_weights = pd.DataFrame({
            'feature': feature_names,
            'weight': component
        })

        # Get top N positive and negative features
        top_positive = pc_weights.sort_values(by='weight', ascending=False).head(num_features_to_show)
        top_negative = pc_weights.sort_values(by='weight', ascending=True).head(num_features_to_show)

        print(f"\n" + "="*70)
        print(f"Rank #{rank}: {pc_name} (Index {pc_index})")
        print("="*70)

        print(f"\n--- Top {num_features_to_show} POSITIVE features ---")
        print(top_positive.to_string())

        print(f"\n--- Top {num_features_to_show} NEGATIVE features ---")
        print(top_negative.to_string())

        # --- Check for Demographic Features ---
        positive_features_list = top_positive['feature'].tolist()
        negative_features_list = top_negative['feature'].tolist()
        
        found_pos = find_demographic_features(positive_features_list, search_terms)
        found_neg = find_demographic_features(negative_features_list, search_terms)

        if found_pos or found_neg:
            log(f"*** Potential Demographic Features Found in {pc_name} (Rank #{rank}) ***")
            if found_pos:
                log(f"  Positive Loadings: {found_pos}")
            if found_neg:
                log(f"  Negative Loadings: {found_neg}")
            # Optional: Stop after the first find if desired
            # break 
        # --- End Check ---

    except Exception as e:
        print(f"\nCould not process {pc_name}: {e}")

log("\nAnalysis finished.")
