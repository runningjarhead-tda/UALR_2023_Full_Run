import pandas as pd
from pathlib import Path

# 1. Load Data
data_path = Path("opafy23nid.parquet") 
df = pd.read_parquet(data_path)

# 2. CLEANING: Force columns to Numeric
df['MONRACE'] = pd.to_numeric(df['MONRACE'], errors='coerce')
df['SENTTOT'] = pd.to_numeric(df['SENTTOT'], errors='coerce')

# Drop missing sentences & Create Copy
df_clean = df.dropna(subset=['SENTTOT']).copy()

# 3. Create Race Label
race_map = {
    1: 'White', 2: 'Black', 3: 'Hispanic', 
    4: 'Other', 5: 'Other', 6: 'Other', 7: 'Other'
}
df_clean['Race_Label'] = df_clean['MONRACE'].map(race_map).fillna('Other')

# 4. EXCLUDE § 2L1.2 (Unlawful Entry)
# We filter it out BEFORE calculating the top 5
df_filtered = df_clean[df_clean['GDLINE1'] != '2L1.2']

# 5. Find the NEW Top 5
top_crimes = df_filtered['GDLINE1'].value_counts().head(5).index
df_subset = df_filtered[df_filtered['GDLINE1'].isin(top_crimes)].copy()

# 6. Add English Descriptions for the Guidelines
# This map covers the most likely crimes to appear in the Top 5
guideline_map = {
    '2D1.1': 'Drug Trafficking',
    '2K2.1': 'Firearms',
    '2B1.1': 'Fraud & Theft',
    '2L1.1': 'Alien Smuggling',
    '2S1.1': 'Money Laundering',
    '2G2.2': 'Child Exploitation',
    '2T1.1': 'Tax Evasion',
    '2A2.2': 'Aggravated Assault'
}

# Create a combined label: "2D1.1 (Drug Trafficking)"
def get_label(code):
    desc = guideline_map.get(code, "Other")
    return f"{code} ({desc})"

df_subset['Guideline_Display'] = df_subset['GDLINE1'].apply(get_label)

# 7. Generate the Table
pivot_table = df_subset.pivot_table(
    index='Guideline_Display',  # Use the new English Label
    columns='Race_Label',       
    values='SENTTOT',           
    aggfunc='mean'              
).round(1)

print("--- Average Sentence Length (Months) by Guideline & Race ---")
print(pivot_table)
