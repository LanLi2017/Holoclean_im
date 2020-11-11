import pandas as pd

df = pd.read_csv('recon_hospital_clean.csv')
dedup_df = df.drop_duplicates()

print(len(dedup_df))