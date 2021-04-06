import pandas as pd
import os
recon_fname= 'android_4_2_KAPPA-Round-1-Reconcile_FIN.csv'
recheck_fname= 'Anns2recheck_4_2.csv'
save_fname= '../../Inputs/FinalizedAnns_4_2_APIs.csv'
'''
Android 4.2
After final reconciliation:
NEITHER    727
SOURCE     145
SINK       128
Name: Chris, dtype: int64
'''
'''
Android API 28-29
After final reconciliation:
NEITHER    750
SOURCE     218
SINK        32
Name: Chris, dtype: int64
'''



print(os.listdir())
df = pd.read_csv(recon_fname, index_col=0)
print("Chris's original annotations:")
print(df['Chris'].value_counts())
print("\nRocky's original annotations:")
print(df['Rocky'].value_counts())

s = df['Chris'].copy()
print('Disagreements:',(df['Agreement']==False).sum())
s[df['Agreement']==False] = df.loc[df['Agreement']==False,'Chris.1']
print("\nAfter first reconciliation:")
print(s.value_counts())
recheck_df = pd.read_csv(recheck_fname, index_col=0)

s[recheck_df.index] = recheck_df['New Annotation']
print("\nAfter final reconciliation:")
print(s.value_counts())
s.to_csv(save_fname)
