import pandas as pd
import math
import re
from utils.SKLearnPrep import make_multi_cols

'''
TODO: Document what this script does
'''



def get_samples_from_each_type(df:pd.DataFrame, n=100, include_na=False):
    samples = []
    for t in df['Source/Sink'].unique():
        if type(t) == str and t.strip() != '' or t != t:
            if t != t and include_na:
                samples.append(df.loc[df['Source/Sink'].isna()].sample(n=n))
            else:
                samples.append(df.loc[df['Source/Sink'] == t].sample(n=n))

    return pd.concat(samples, axis=0)



df = pd.read_pickle("../../Inputs/Caches/cache.pickle")
chris = pd.read_csv("to_be_annotated/chris_anns.csv", index_col=0)

dsafe = df.loc[df['Origin']=='dsafe']#.sample(n=100)

chris['Origin'] ='chris'
chris = chris.drop('Cat GroundT', axis=1)
chris['Source/Sink'] = chris['SS GroundT']
print(chris.shape)
print(chris['Source/Sink'].unique())
# chris_anned = chris.loc[chris['Source/Sink'].notna()].sample(n=75)
# chris_unanned = chris.loc[chris['Source/Sink'].isna()].sample(n=50)
# chris = chris_anned.append(chris_unanned)
chris = chris.drop('SS GroundT', axis =1)
col_order = ['Source/Sink', 'Origin','Description', 'Return', 'Parameters', 'ApiLevel']
chris = chris[col_order]
chris = get_samples_from_each_type(chris, n=8, include_na=True)
chris['Source/Sink'] = chris['Source/Sink'].str.replace("1", 'source')
chris['Source/Sink'] = chris['Source/Sink'].str.replace("2", 'sink')
chris['Source/Sink'] = chris['Source/Sink'].str.replace("3", 'none')

dsafe = make_multi_cols(dsafe)
dsafe = dsafe.drop('ManFeatures', axis=1, level=0)
dsafe.columns = dsafe.columns.get_level_values(1)
dsafe = dsafe.drop(['QualifiedPackage', 'Classname', 'MethodName'], axis=1)
dsafe = dsafe.loc[dsafe['Description'] != '']
dsafe = dsafe[col_order]
dsafe = get_samples_from_each_type(dsafe, n=8)
dsafe.to_csv('ann_samples/dsafe_sample.csv')
chris.to_csv('ann_samples/chris_sample.csv')
print(dsafe['Source/Sink'].value_counts())
print(chris['Source/Sink'].value_counts())
print(chris.columns)
print(dsafe.columns)

