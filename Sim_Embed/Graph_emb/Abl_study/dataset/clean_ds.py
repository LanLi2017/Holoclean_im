import pandas as pd
fp = 'hospital_100.csv'
df = pd.read_csv(fp, index_col=False)
df_clean = df.drop(columns=['Address2','Address3', 'State','HospitalType','HospitalOwner',
                            'EmergencyService','Condition','MeasureCode','MeasureName',
                            'Score','Sample','Stateavg'])
modDf = df_clean.dropna()
modDf.to_csv('med_demo.csv', index=False)