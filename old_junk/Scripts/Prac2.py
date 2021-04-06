import pandas as pd
import numpy as np
from utils.FormattingTools import Preprocessor

docs_fname = "Inputs/android.csv"
p = Preprocessor()
names = ['QualifiedPackage', 'Classname', 'MethodName', 'Description', 'Parameters', 'Return', 'ApiLevel']
docs = pd.read_csv(docs_fname, names=names)
df = p.formatDocs(docs)

for col in df.columns:
    print(df[col].dtype)
    if df[col].dtype == np.object:
        df[col] = df[col].str.replace(r"\r\n", ' ', regex=True)
        df[col] = df[col].str.replace(r"\n", ' ', regex=True)
        df[col] = df[col].str.replace(r"\\r\\n", ' ', regex=True)
        df[col] = df[col].str.replace(r"\\n", ' ', regex=True)
df.to_csv('formatted_docs.csv',header=False)
# c_df = pd.read_csv("to_be_annotated/chris_anns.csv", index_col=0)
# # c_df = c_df.loc[(c_df["SS GroundT"] ==1) | (c_df["SS GroundT"] ==2)]
# m_df = pd.read_csv("Inputs/mitra_anns2.csv", index_col=0)
# m_df = m_df.loc[(m_df["SS GroundT"] ==1) | (m_df["SS GroundT"] ==2)]
# r_df = pd.read_csv("Inputs/rocky_anns2.csv", index_col=0)
# # r_df = r_df.loc[(r_df["SS GroundT"] ==1) | (r_df["SS GroundT"] ==2)]
# m_df = pd.concat([m_df, c_df, r_df], axis=0)
# m_df['SS GroundT']= pd.to_numeric(m_df['SS GroundT'], errors="coerce")
# print(m_df['SS GroundT'].notna().sum())
# m_df= m_df.loc[m_df['SS GroundT'].notna()].sample(n=m_df['SS GroundT'].notna().sum())
# print(m_df.columns)
# print(m_df['SS GroundT'].value_counts())
# m_df.loc[ m_df['SS GroundT']==1,'SS GroundT'] ='source'
# m_df.loc[ m_df['SS GroundT']==2,'SS GroundT'] ='sink'
# m_df.loc[m_df['SS GroundT']==3,'SS GroundT'] ='none'
# print(m_df['SS GroundT'].value_counts()/m_df['SS GroundT'].notna().sum())
# print(m_df['SS GroundT'].value_counts())
# print(m_df)
# m_df.to_csv("Inputs/new_anns2.csv")
# df = pd.read_csv("Inputs/lissa_annotations.csv", index_col=0,names=["Origin", "SS", "Cat"])
# # print(df['Origin'].value_counts())
# print("DSAFE")
# print(df.loc[df['Origin']=='dsafe', 'SS'].value_counts())
# df = pd.read_pickle("cache.pickle")
# # print(df.columns)
# dsafe = df.loc[(df['Origin']=='dsafe') & (df["ApiLevel"] <= 19)]
# print("pct in 19")
# pct19 = (dsafe["ApiLevel"]==19).sum()/ dsafe.shape[0]
# print(pct19, "pct under 19", 1-pct19)
