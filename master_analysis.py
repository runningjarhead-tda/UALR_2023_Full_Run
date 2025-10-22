# -*- coding: utf-8 -*-
# FINAL PRODUCTION SCRIPT (All parts active, corrected Scree Plot)

# ================== 1. IMPORTS and SETUP ==================
%matplotlib inline
import os, re, sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import cudf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")

# Safer threading
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from cuml.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost
import shap

# ================== 2. CONFIGURATION ==================
SEED = 42
K_CLUSTERS = 4
TARGET_CHOICES = ["SENTTOT", "TOTSENTN", "SENTENCE_MONTHS"]
MAX_PCA_COMPONENTS = 50

# ACCURATE RACE MAPPING based on USSC Codebook for MONRACE
RACE_CODE_MAP = {
    1: "White",
    2: "Black",
    3: "American Indian/Alaskan Native",
    4: "Asian or Pacific Islander",
    5: "Multi-racial",
    7: "Other",
    8: "Not Available",
    9: "Non-US American Indians",
    10: "American Indians Citizenship Unknown"
}

OFFENSE_CODE_MAP = {
    1: "Murder", 2: "Manslaughter", 3: "Kidnapping/Hostage Taking", 4: "Sexual Abuse",
    5: "Assault", 6: "Robbery", 9: "Arson", 10: "Drugs - Trafficking",
    11: "Drugs - Communication", 12: "Drugs - Simple Possession", 13: "Firearms",
    15: "Burglary/Breaking & Entering", 16: "Auto Theft", 17: "Larceny", 18: "Fraud",
    19: "Embezzlement", 20: "Forgery/Counterfeiting", 21: "Bribery", 22: "Tax Offenses",
    23: "Money Laundering", 24: "Racketeering/Extortion", 25: "Gambling/Lottery",
    26: "Civil Rights Offenses", 27: "Immigration", 28: "Pornography/Prostitution",
    29: "Prison Offenses", 30: "Administration of Justice",
    31: "Environmental/Wildlife", 32: "National Defense", 33: "Antitrust Violations",
    34: "Food and Drug Offenses", 35: "Traffic & Other Offenses",
    42: "Child Pornography", 43: "Obscenity", 44: "Prostitution"
}

# ================== 3. HELPER FUNCTIONS ==================
def log(s): print(f"[POC] {s}", flush=True)
def _norm(s): return re.sub(r"[\W_]+","",str(s).strip().upper())
def _ensure_dir(p): p=Path(p); p.mkdir(parents=True, exist_ok=True); return p

def smart_read(path):
    p = Path(path)
    if p.suffix.lower() == ".parquet": return pd.read_parquet(p)
    if p.suffix.lower() == ".csv": return pd.read_csv(p)
    raise ValueError("Provide .parquet or .csv")

def first_existing(df, names):
    for n in names:
        if n in df.columns: return n
        for col in df.columns:
            if _norm(col) == _norm(n): return col
    return None

def find_offense_column(df):
    best_candidate, highest_score = None, -1
    for col_name in df.columns:
        score = 0; norm_name = _norm(col_name)
        if "OFF" in norm_name: score += 2
        if any(s in norm_name for s in ["TYPE", "GUIDE", "CAT"]): score += 1
        # Check if column is numeric or can be coerced
        try:
            numeric_col = pd.to_numeric(df[col_name], errors='coerce')
            if numeric_col.notna().any(): # Check if at least one value is numeric
                valid_codes = numeric_col.isin(OFFENSE_CODE_MAP.keys()).sum()
                total_non_null = numeric_col.notna().sum()
                if total_non_null > 0 and (valid_codes / total_non_null) > 0.7: score += 5
        except Exception:
            pass # Not a numeric or coercible column

        if score > highest_score:
            highest_score, best_candidate = score, col_name

    if best_candidate: log(f"Found most likely offense column: '{best_candidate}'.")
    else: log("Warning: Could not automatically identify an offense code column.")
    return best_candidate

def map_race_ethnicity(df):
    # This helper no longer needs the log() call, it's just a mapper
    if 'NEWRACE' in df.columns:
        newrace_map = {1: "White", 2: "Black", 3: "Hispanic", 4: "Asian", 5: "American Indian/Alaskan Native", 6: "Other"}
        return pd.to_numeric(df['NEWRACE'], errors='coerce').map(newrace_map)

    race_col = first_existing(df, ["MONRACE", "RACE"])
    hisp_col = first_existing(df, ["HISPORIG"])
    if not race_col or not hisp_col: return pd.Series(index=df.index)

    race = pd.to_numeric(df[race_col], errors='coerce').map(RACE_CODE_MAP)
    is_hispanic = pd.to_numeric(df[hisp_col], errors='coerce') == 1
    return pd.Series(np.where(is_hispanic, "Hispanic", race), index=df.index)

def map_gender(df):
    gender_col = first_existing(df, ["MONSEX", "SEX", "GENDER"])
    if not gender_col: return pd.Series(index=df.index)
    codes = pd.to_numeric(df[gender_col], errors='coerce')
    return codes.map({0: "Male", 1: "Female"})

def map_offense_type(df):
    offense_col = find_offense_column(df)
    if not offense_col: return pd.Series(index=df.index)
    # This is the key: always coerce to numeric before mapping
    codes = pd.to_numeric(df[offense_col], errors='coerce')
    return codes.map(OFFENSE_CODE_MAP)

# ================== 4. DEMOGRAPHIC TABLE FUNCTION ==================
def generate_demographic_breakdown(df, target_col, out_dir):
    log("--- Generating Final, Unfiltered Demographic Breakdown with All Case Outcomes ---")

    # Create a copy to avoid modifying the original df
    df_demo = df.copy()

    # Clean column headers of any hidden whitespace
    df_demo.columns = df_demo.columns.str.strip()

    # --- Map Demographic Columns ---
    log("Mapping offense, race, and gender...")
    df_demo['OffenseCategory'] = map_offense_type(df_demo).fillna('Unknown/Missing')
    df_demo['RaceEthnicity'] = map_race_ethnicity(df_demo).fillna('Unknown')
    df_demo['GenderName'] = map_gender(df_demo).fillna('Unknown')

    df_demo[target_col] = pd.to_numeric(df_demo[target_col], errors='coerce')

    # --- Process Case Disposition with correct codes ---
    log("Processing case disposition data...")
    disposit_col = first_existing(df_demo, ["DISPOSIT"])
    if disposit_col:
        disposition_map = {
            '1': 'Conviction', '2': 'Conviction',
            '3': 'Dismissed', '4': 'Acquitted'
        }
        df_demo['CaseOutcome'] = df_demo[disposit_col].map(disposition_map).fillna('Other')
        df_demo['Count_Conviction'] = (df_demo['CaseOutcome'] == 'Conviction').astype(int)
        df_demo['Count_Dismissed'] = (df_demo['CaseOutcome'] == 'Dismissed').astype(int)
        df_demo['Count_Acquitted'] = (df_demo['CaseOutcome'] == 'Acquitted').astype(int)
        df_demo['Count_Other'] = (df_demo['CaseOutcome'] == 'Other').astype(int)
    else:
        log("Warning: 'DISPOSIT' column not found. Skipping case outcome calculations.")
        df_demo['Count_Conviction'], df_demo['Count_Dismissed'], df_demo['Count_Acquitted'], df_demo['Count_Other'] = [0, 0, 0, 0]

    # --- Process Supervised Release Data ---
    log("Processing supervised release data...")
    suprdum_col = first_existing(df_demo, ["SUPRDUM"])
    suprrel_col = first_existing(df_demo, ["PROBATN"])
    if suprdum_col and suprrel_col:
        df_demo[suprdum_col] = pd.to_numeric(df_demo[suprdum_col], errors='coerce').fillna(0)
        df_demo[suprrel_col] = pd.to_numeric(df_demo[suprrel_col], errors='coerce').fillna(0)
        numeric_suprrel = df_demo[suprrel_col].replace([996, 997], np.nan)
        df_demo['SupervisedReleaseLength'] = np.where(df_demo[suprdum_col] == 1, numeric_suprrel, np.nan)
        df_demo['SupervisedReleaseLifeCount'] = ((df_demo[suprdum_col] == 1) & (df_demo[suprrel_col] == 996)).astype(int)
        df_demo['SupervisedReleaseNotSpecCount'] = ((df_demo[suprdum_col] == 1) & (df_demo[suprrel_col] == 997)).astype(int)
    else:
        df_demo['suprdum'], df_demo['SupervisedReleaseLength'], df_demo['SupervisedReleaseLifeCount'], df_demo['SupervisedReleaseNotSpecCount'] = [0, np.nan, 0, 0]

    # --- Process Restitution Data ---
    log("Processing restitution data...")
    restdum_col = first_existing(df_demo, ["RESTDUM"])
    restitution_amt_col = first_existing(df_demo, ["TOTREST", "AMTTOTAL"])
    if restdum_col and restitution_amt_col:
        df_demo[restdum_col] = pd.to_numeric(df_demo[restdum_col], errors='coerce').fillna(0)
        df_demo[restitution_amt_col] = pd.to_numeric(df_demo[restitution_amt_col], errors='coerce')
        df_demo['RestitutionAmount'] = np.where(df_demo[restdum_col] == 1, df_demo[restitution_amt_col], np.nan)
    else:
        df_demo['restdum'], df_demo['RestitutionAmount'] = [0, np.nan]

    # --- Define Aggregation Logic ---
    suprdum_agg_col = suprdum_col if suprdum_col else 'suprdum'
    restdum_agg_col = restdum_col if restdum_col else 'restdum'
    agg_funcs = {
        'Count_Conviction': ['sum'], 'Count_Dismissed': ['sum'],
        'Count_Acquitted': ['sum'], 'Count_Other': ['sum'],
        'SENTTOT': ['mean', 'median'],
        suprdum_agg_col: ['sum'], 'SupervisedReleaseLength': ['mean', 'median'],
        'SupervisedReleaseLifeCount': ['sum'], 'SupervisedReleaseNotSpecCount': ['sum'],
        restdum_agg_col: ['sum'], 'RestitutionAmount': ['mean', 'median']
    }

    # --- Perform Grouping and Aggregation (dropna=False) ---
    log("Aggregating data...")
    breakdown = df_demo.groupby(['OffenseCategory', 'RaceEthnicity', 'GenderName'], dropna=False).agg(agg_funcs)

    breakdown.columns = [
        'ConvictionCount', 'DismissalCount', 'AcquittalCount', 'OtherOutcomeCount',
        'MeanSentence', 'MedianSentence',
        'SupervisedReleaseCount', 'MeanSupervisedRelease_Months', 'MedianSupervisedRelease_Months',
        'Count_SR_Life', 'Count_SR_NotSpecified',
        'RestitutionCount', 'MeanRestitution_USD', 'MedianRestitution_USD'
    ]

    outcome_cols = ['ConvictionCount', 'DismissalCount', 'AcquittalCount', 'OtherOutcomeCount']
    breakdown['TotalAllOutcomes'] = breakdown[outcome_cols].sum(axis=1)

    for col in ['MeanSentence', 'MedianSentence', 'MeanSupervisedRelease_Months', 'MedianSupervisedRelease_Months']:
        if col in breakdown.columns: breakdown[col] = breakdown[col].round(1)
    for col in ['MeanRestitution_USD', 'MedianRestitution_USD']:
        if col in breakdown.columns: breakdown[col] = breakdown[col].round(2)

    breakdown.index.set_names(['OffenseCategory', 'RaceEthnicity', 'GenderName'], inplace=True)
    breakdown.reset_index(inplace=True)
    breakdown.fillna({'OffenseCategory': 'Unknown/Missing', 'RaceEthnicity': 'Unknown', 'GenderName': 'Unknown'}, inplace=True)

    # --- Save Table 1 from Paper (CORRECTED) ---
    log("Generating offense category distribution (Table 1)...")
    if 'OffenseCategory' in df_demo.columns:
        offense_counts = df_demo['OffenseCategory'].value_counts(normalize=True).reset_index()
        offense_counts.columns = ['OffenseCategory', 'Percentage']
        offense_counts['Percentage'] = (offense_counts['Percentage'] * 100).round(2)
        offense_table_path = Path(out_dir) / "Table_1_Offense_Category_Distribution.csv"
        offense_counts.to_csv(offense_table_path, index=False)
        log(f"Saved offense distribution (Table 1) to: {offense_table_path}")
    else:
        log("Warning: 'OffenseCategory' column not found. Skipping Table 1 generation.")

    # --- Save Main Demographic Table ---
    output_path = Path(out_dir) / "demographic_breakdown_table.tsv"
    breakdown.to_csv(output_path, sep='\t', index=False)
    log(f"Saved full demographic breakdown to: {output_path}")

    # Print a snippet to console
    pd.set_option('display.max_rows', 500)
    print(breakdown.head(10).to_string())
    log("----------------------------------------------------------------")
    return breakdown

# ================== 5. PCA/XAI ANALYSIS FUNCTION ==================
def tda_xai_full(df, out_dir, target_col):
    out_dir = _ensure_dir(out_dir)
    cache_prefix = "full_dataset"

    df_analysis = df.copy()
    df_analysis.columns = df_analysis.columns.str.strip() # Clean column names

    log("Running full data prep and PCA.")

    df_clean = df_analysis.dropna(subset=[target_col]).copy()
    df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors="coerce")
    y = df_clean[target_col].astype(float).values

    df_clean['OffenseCategory'] = map_offense_type(df_clean).fillna('Unknown/Missing')

    potential_feature_cols = [c for c in df_clean.columns if c != target_col and 'ID' not in c.upper() and all(s not in c for s in ['OffenseCategory', 'RaceEthnicity', 'GenderName', 'CaseOutcome'])]
    numeric_features = df_clean[potential_feature_cols].select_dtypes(include=np.number).columns.tolist()
    categorical_features = df_clean[potential_feature_cols].select_dtypes(exclude=np.number).columns.tolist()

    X_num = df_clean[numeric_features].apply(lambda x: x.fillna(x.median()), axis=0)

    X_cat_parts = []
    log(f"Starting one-hot encoding for {len(categorical_features)} categorical features...")
    for col in categorical_features:
        top_50 = df_clean[col].value_counts().head(50).index
        df_clean[col] = df_clean[col].where(df_clean[col].isin(top_50), 'Other')
        dummies = pd.get_dummies(df_clean[col], prefix=col, drop_first=False, dtype=np.uint8)
        X_cat_parts.append(dummies)

    X_cat = pd.concat(X_cat_parts, axis=1)
    X = pd.concat([X_num, X_cat], axis=1).fillna(0)
    feature_names = X.columns.tolist()

    log(f"Created feature matrix with {X.shape[0]} samples and {X.shape[1]} features.")

    n_components = MAX_PCA_COMPONENTS

    log(f"Using {n_components} PCA components.")
    scaler = StandardScaler()
    pca = IncrementalPCA(n_components=n_components, batch_size=5000)

    log("Fitting scaler and PCA in batches...")
    for i in range(0, X.shape[0], 5000):
        batch = X.iloc[i:i+5000]
        batch_scaled = scaler.fit_transform(batch)
        pca.partial_fit(batch_scaled)

    log("Transforming full dataset in batches...")
    lens_parts = []
    for i in range(0, X.shape[0], 5000):
        batch = X.iloc[i:i+5000]
        batch_scaled = scaler.transform(batch)
        lens_parts.append(pca.transform(batch_scaled))
    lens = np.concatenate(lens_parts, axis=0)

    log("PCA complete.")

    # *** SAVE THE MODELS FOR INTERPRETATION ***
    log("Saving PCA model, scaler, and feature names...")
    joblib.dump(pca, out_dir / "pca_model.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    with open(out_dir / "feature_names.json", 'w') as f:
        json.dump(feature_names, f)
    log(f"Saved models to {out_dir}")
    # *** END BLOCK ***

    # --- Generate Figure 1: Scree Plot (Individual Component Variance with Labels) ---
    plt.figure(figsize=(12, 7)) # Increased size for readability

    variances = pca.explained_variance_ratio_
    component_numbers = range(1, n_components + 1)

    # Create the bar plot
    bars = plt.bar(component_numbers, variances, alpha=0.8, align='center', label='Individual Explained Variance')

    # Add text labels (percentages) above each bar
    for bar in bars:
        yval = bar.get_height()
        # Format as percentage with 1 decimal place
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.001, f'{yval*100:.1f}%', va='bottom', ha='center', fontsize=8, rotation=90) # Added rotation

    # Optional: Keep the elbow curve for visual trend
    plt.plot(component_numbers, variances, 'o-', color='red', markersize=3, linewidth=1, label='Elbow Curve')

    plt.xlabel('Principal Component Number')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot: Explained Variance by Component with Percentage Labels')
    plt.xticks(range(0, n_components + 1, 5)) # Show ticks every 5 components
    plt.ylim(0, variances[0] * 1.15) # Adjust y-limit to make space for labels
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout() # Adjust layout

    fig1_path = out_dir / "Figure_1_Scree_Plot_Individual_Labeled.png" # Updated filename
    plt.savefig(fig1_path)
    plt.close()
    log(f"Saved Labeled Individual Scree Plot (Figure 1) to: {fig1_path}")
    log(f"Total variance explained by {n_components} components: {np.sum(variances):.3f}")

    # --- XGBoost Modeling ---
    log("Starting XGBoost training...")
    pc_names = [f'PC_{i+1}' for i in range(n_components)]
    lens_df = pd.DataFrame(lens, columns=pc_names)

    Xtr, Xte, ytr, yte = train_test_split(lens_df, y, test_size=0.2, random_state=SEED)

    model = xgboost.XGBRegressor(n_estimators=300, random_state=SEED, tree_method='gpu_hist', objective='reg:squarederror')
    model.fit(Xtr, ytr)
    r2 = float(r2_score(yte, model.predict(Xte)))
    log(f"XGBoost R² = {r2:.3f}")

    # --- SHAP Analysis ---
    log("Starting SHAP analysis...")
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(Xte)

    shap_df = pd.DataFrame(sv, columns=pc_names)

    # --- Generate Figure 2: SHAP Summary Plot (Beeswarm) ---
    log("Generating SHAP beeswarm plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, Xte, show=False, max_display=20)
    plt.gcf().suptitle("SHAP Summary Plot: Impact on Sentence Length", fontsize=14)
    fig2_path = out_dir / "Figure_2_SHAP_Summary_Beeswarm_Plot.png"
    plt.savefig(fig2_path, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved SHAP beeswarm plot (Figure 2) to: {fig2_path}")

    # --- Generate Table 2: SHAP Importance ---
    shap_importance = pd.DataFrame(shap_df.abs().mean(), columns=['Mean_Absolute_SHAP'])
    shap_importance = shap_importance.sort_values(by='Mean_Absolute_SHAP', ascending=False)
    table2_path = out_dir / "Table_2_SHAP_Importance.csv"
    shap_importance.to_csv(table2_path)
    log(f"Saved SHAP importance (Table 2) to: {table2_path}")

    # --- Generate Figure 3: PC Scores by Crime Boxplot ---
    lens_df['OffenseCategory'] = df_clean['OffenseCategory'].values
    top_5_pcs = shap_importance.head(5).index.tolist()

    top_10_offenses = df_clean['OffenseCategory'].value_counts().head(10).index
    plot_data = lens_df[lens_df['OffenseCategory'].isin(top_10_offenses)]

    fig, axes = plt.subplots(len(top_5_pcs), 1, figsize=(15, 5 * len(top_5_pcs)), sharex=True)
    fig.suptitle('Distribution of Top 5 PC Scores by Offense Category', fontsize=16)
    for i, pc in enumerate(top_5_pcs):
        sns.boxplot(ax=axes[i], x='OffenseCategory', y=pc, data=plot_data)
        axes[i].set_title(f'Feature Importance Rank {i+1}: {pc}')
        axes[i].set_ylabel('PC Score')
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig3_path = out_dir / "Figure_3_PC_Scores_by_Offense.png"
    plt.savefig(fig3_path)
    plt.close()
    log(f"Saved PC score boxplots (Figure 3) to: {fig3_path}")

    log("--- Full Analysis Complete ---")
    return {
        "r2": r2, "model": model, "pca": pca,
        "feature_names": feature_names, "shap_importance": shap_importance
    }


# ================== 6. MAIN RUNNER FUNCTION (CORRECTED) ==================
def run_all(input_path, out_dir="."):
    out_dir = _ensure_dir(out_dir)
    df = smart_read(input_path)
    log(f"Loaded {len(df):,} rows")

    target_col = first_existing(df, TARGET_CHOICES)
    if not target_col:
        log(f"CRITICAL ERROR: Could not find target variable. Searched for: {TARGET_CHOICES}")
        return

    # --- PART 1: Generate Demographic TSV Table ---
    log("\n--- Starting Part 1: Demographic Table Generation ---")
    generate_demographic_breakdown(df, target_col, out_dir)

    # --- PART 2: Generate Journal Paper Figures & Analysis ---
    # *** THIS IS THE FIX: This line is now UN-COMMENTED ***
    log("\n--- Starting Part 2: Full PCA/XAI Analysis for Journal Paper ---")
    full_run_results = tda_xai_full(df.copy(), out_dir, target_col=target_col)

    if full_run_results:
        log(f"\n--- Analysis Summary ---")
        log(f"Final R-squared: {full_run_results['r2']:.4f}")
        log("Top 5 Most Important Principal Components (from SHAP):")
        print(full_run_results['shap_importance'].head(5).to_string())

# ================== 7. EXECUTION CELL FOR JUPYTER ==================
input_path = "/content/opafy23nid.parquet"
output_folder = "fy2023_analysis_results"

log("Starting full analysis...")
run_all(input_path=input_path, out_dir=output_folder)
log(f"\nAnalysis complete. All results saved to the '{output_folder}' directory.")
