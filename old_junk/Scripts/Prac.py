import pandas as pd
import re
from utils.SKLearnPrep import make_multi_cols
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

df = pd.read_pickle("../../Inputs/Caches/cache.pickle")
print("num before", df.index.size)
df = df.loc[~df.index.duplicated(keep='first')]
print("num after", df.index.size)
# print(df["Origin"].unique())
# print("susi")
# print(df.loc[df['Origin']=='susi'].index.size)
# print(df.loc[df['Origin']=='dsafe'].index.size)
# print(df.loc[df["Classname"].notna() ].index.size)
# print(df.loc[df["Classname"].notna() & df["Origin"].notna()].index.size)
#
# with open("permissionMethodWithLabel.pscout") as f:
#     cnt = 0
#     for line in f.readlines():
#         mo = re.search(r"_SOURCE_", line)
#         if mo is not None:
#             cnt += 1
#             print(line.strip())
#     print(cnt)

new_apis = df.loc[(df['ApiLevel'] >= 28) & df["Classname"].notna()].copy()
new_apis["SS GroundT"] = ""
new_apis["Cat GroundT"] = ''
new_apis = new_apis[["SS GroundT","Cat GroundT", "Description", "Return", 'Parameters', 'ApiLevel']].sample(frac=1.0)
split_size = new_apis.index.size//3
print(new_apis.columns)
chris = new_apis.iloc[:split_size]
mitra = new_apis.iloc[split_size:2*split_size]
rocky = new_apis.iloc[2*split_size:]
# chris.to_csv("chris_anns.csv")
# mitra.to_csv("mitra_anns2.csv")
# rocky.to_csv("rocky_anns2.csv")



contemperary = df.loc[(df['ApiLevel'] < 21) & (df['ApiLevel']>0)& df["Origin"].notna()]
# print("meths w/ docs")
# print(contemperary["Classname"].notna().sum()/contemperary.index.size)
# print(df.loc[(df['ApiLevel'] < 21)].index.size)
# dsafe = contemperary.loc[contemperary['Origin']=='dsafe']
# print("none")
# print((dsafe["Source/Sink"]=='none').sum()/dsafe["Source/Sink"].index.size)
# print("source")
# print((dsafe["Source/Sink"]=='source').sum()/dsafe["Source/Sink"].index.size)
# print("sink")
# print((dsafe["Source/Sink"]=='sink').sum()/dsafe["Source/Sink"].index.size)

# print("Network anns ")
# print(contemperary.loc[(contemperary["Origin"]=='susi')& (contemperary["Category"] == 'NETWORK')].index.size)
# print(contemperary.loc[(contemperary["Origin"]=='dsafe')& (contemperary["Category"] == 'NETWORK')].index.size)
# print("dsafe")
# print(contemperary.loc[(contemperary["Origin"]=='dsafe')].index.size)
# print(contemperary.loc[(contemperary["Origin"]=='dsafe')& (contemperary["Source/Sink"] == 'source')].index.size)
# print(contemperary.loc[(contemperary["Origin"]=='dsafe')& (contemperary["Source/Sink"] == 'sink')].index.size)
# print("susi")
# print(contemperary.loc[(contemperary["Origin"]=='susi')].index.size)
# print(contemperary.loc[(contemperary["Origin"]=='susi')& (contemperary["Source/Sink"] == 'source')].index.size)
# print(contemperary.loc[(contemperary["Origin"]=='susi')& (contemperary["Source/Sink"] == 'sink')].index.size)
# susi_df = contemperary.loc[(contemperary["Origin"]=='susi')]
# dsafe_df = contemperary.loc[(contemperary["Origin"]=='dsafe')]
# susi_df = make_multi_cols(susi_df)
# dsafe_df = make_multi_cols(dsafe_df)
# # dsafe_df["DocFeatures"].to_csv("dsafe_anns.csv")
# # print(susi_df["DocFeatures"]["Source/Sink"].value_counts())
# # print(df.loc[df["Origin"]=='susi']["Source/Sink"].value_counts())
# # print("DROIDSAFE--------------")
# # print(dsafe_df["DocFeatures"]["Source/Sink"].value_counts())
# # print(df.loc[df["Origin"]=='dsafe']["Source/Sink"].value_counts())
# print(df["Source/Sink"].value_counts())
# print(df.index.size)
# print(df["Source/Sink"].isna().sum())
# df = df.loc[(df['ApiLevel'] < 21) & (df['ApiLevel']>0)]
# print("Documented")
# print(df["Source/Sink"].value_counts())
# print(df.index.size)
# print(df["Source/Sink"].isna().sum())
# print("Documented w/ na")
# df["Source/Sink"][df["Source/Sink"].isna()] = 'none'
# print(df["Source/Sink"].value_counts())
# print(df.index.size)
# print(df["Source/Sink"].isna().sum())



#Make fig 1
plt.style.use('seaborn-pastel')
plt.rcParams.update({'font.size': 12})
df = df["ApiLevel"][df["ApiLevel"].notna()&(df["ApiLevel"]>1.0)].value_counts().sort_index()
print(df.to_numpy())
print(df.index.to_series().to_numpy())
plt.bar(df.index.to_series().to_numpy(), df.to_numpy())
lr = Ridge()
# print(df.to_numpy())
lr.fit(df.index.to_series().to_numpy().reshape(-1,1),df.to_numpy().reshape(-1,1))
plt.plot(df.index.to_series().to_numpy(), (lr.coef_*df.index.to_series().to_numpy()+lr.intercept_).flatten(),color='orange')
print(df.index.to_series().to_numpy().reshape(1,-1))
print(lr.coef_*df.index.to_series().to_numpy()+lr.intercept_)
plt.ylabel("#Methods")
plt.xlabel("API version")
# plt.title("Number of Methods In Android API")
plt.show()



