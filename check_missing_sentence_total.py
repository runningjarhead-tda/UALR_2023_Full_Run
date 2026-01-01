import pandas as pd

df = pd.read_parquet("/workspace/opafy23nid.parquet")

# 1. Start with the Total
total_records = len(df)
print(f"Total Records in File:      {total_records}")  # Should be 64,124

# 2. Identify the rows to drop (Subset of the Total)
# We are looking inside the existing 64,124 records
missing_sent_count = df['SENTTOT'].isna().sum()
print(f"Rows with Missing SENTTOT: -{missing_sent_count}") # Should be 4,857

# 3. Calculate the Remainder
final_count = total_records - missing_sent_count
print(f"Records Remaining:          {final_count}")     # Should be 59,267
