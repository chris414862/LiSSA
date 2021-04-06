##### Script to convert annotation signitures to the fully qualified sigs in from manual feature extraction #######



# ###### Find SuSi weights with no expression in in-feature-domain data
# import pandas as pd
#
# susi_weights = pd.read_csv('top_susi_weights.csv', index_col=0)
# null_ifd_feats = pd.read_csv('in_feat_domain_no_feats.csv', index_col=0)
# susi_weights.index = susi_weights.index.str.replace(r"<<.*>>", "")
# pd.set_option('display.max_row', 1000)
# print(susi_weights)
# print(susi_weights.shape)
# print("\n")
# print(null_ifd_feats)
#
# print(susi_weights.index.isin(null_ifd_feats.index).sum())
# print(susi_weights.index[susi_weights.index.isin(null_ifd_feats.index)])
#
# doc_man_feats = pd.read_pickle("Inputs/Caches/DocOnlyManFeats.pickle")
# # print(doc_man_feats.loc[doc_man_feats.index.str.contains("ContentR"),'<Parameter type contains android.content.contentresolver>'])


# ######Make fig 1
# from utils.FormattingTools import Preprocessor
# import matplotlib.pyplot as plt
# from sklearn.linear_model import Ridge
#
#
# p = Preprocessor(docs_fname='Inputs/android_30.csv')
# df = p.preprocess_data()
#
# plt.style.use('seaborn-pastel')
# plt.rcParams.update({'font.size': 12})
# fig = plt.figure(figsize=(5,10))
# ax1= fig.add_subplot(211)
# s = df["ApiLevel"].value_counts().sort_index().cumsum()
# print(s)
# s = s[s.index > 1]
# ax1.set_ylabel("#Total Methods")
# ax1.set_xlabel("API Version")
# ax1.plot(s.index.to_series().to_numpy(), s.to_numpy())
# s = df.loc[df["ApiLevel"].notna()&(df["ApiLevel"]>1.0),"ApiLevel"].value_counts().sort_index()
# ax2 = fig.add_subplot(212)
# s = df.loc[df["ApiLevel"].notna()&(df["ApiLevel"]>1.0),"ApiLevel"].value_counts().sort_index()
# ax2.bar(s.index.to_series().to_numpy(), s.to_numpy())
# lr = Ridge()
# lr.fit(s.index.to_series().to_numpy().reshape(-1,1),s.to_numpy().reshape(-1,1))
# y = (lr.coef_*s.index.to_series().to_numpy()+lr.intercept_).flatten()
# ax2.plot(s.index.to_series().to_numpy(), y ,color='orange')
# ax2.set_ylabel("#Methods added")
# ax2.set_xlabel("API Version")
# plt.tight_layout()
# plt.show()


# ############ # Dataset statistics
#
# import pandas as pd
# fname = 'Annotations/chris_new_apis_sample.csv'
# fname = 'Inputs/FinalizedAnns_4_2_APIs.csv'
#
# ann_col = 'Annotation'
# # ann_col = 'Source?'
# # ann_col = 'Sink?'
# print(pd.read_csv(fname, index_col=0))
# print(pd.read_csv(fname, index_col=0).columns)
# df_counts = pd.read_csv(fname, index_col=0)[ann_col].value_counts()
# df_counts = df_counts.reindex(['SOURCE','SINK','NEITHER'])
# df_counts.index = ['Source', 'Sink', 'Neither']
# print(df_counts)
# print(df_counts.to_latex())


# # Aggregates text files of SuSi's output and averages them
# import os
# import re
# import pandas as pd
#
# exp_dir = 'SuSiResults_exp3'
# df_sum = None
# for filename in os.listdir(exp_dir):
#     with open(exp_dir+'/'+filename) as f:
#         lines = [re.split(r"  +",line.strip()) for line in f.readlines() ]
#
#     col_names = lines.pop(0)
#     cols = [zipped for zipped in zip(*lines)]
#     data = {col_name:col for col_name, col in zip(col_names, cols) if col_name not in ['TP Rate', 'FP Rate', 'ROC Area']}
#
#     df = pd.DataFrame(data, index=data.pop('Class'), dtype=float)
#
#     if df_sum is None:
#         df_sum = df
#     else:
#         df_sum += df
#
# df_avg:pd.DataFrame = df_sum/len(os.listdir(exp_dir))
# classes:pd.DataFrame = df_avg.loc[['source', 'sink', 'neithernor']]
# print(pd.DataFrame(classes.mean(axis=0), columns=['Macro Avg.']).transpose())
# df_avg = df_avg.append(pd.DataFrame(classes.mean(axis=0), columns=['Macro Avg.']).transpose())
# df_avg = df_avg.reindex(['source', 'sink', 'neithernor', 'Macro Avg.', 'Weighted Avg.'])
# df_avg.index = ['Source', 'Sink', 'Neither', 'Macro Avg.', 'Weighted Avg.']
# df_avg.columns = ['Precision', 'Recall', 'F1']
# print(df_avg)
# print(df_avg.round(3).to_latex())


# ######################
# ### Format annotations for SuSi
# import csv
#
# ann_fname = "Inputs/FinalizedAnns_4_2_APIs.csv"
# output_fname = "FA_4_2_APIs.pscout"
# lines = []
# with open(ann_fname)as f:
#     my_rdr = csv.reader(f)
#     for i,row in enumerate(my_rdr):
#         if i == 0:
#             continue
#         sig, ann = row[0], row[1]
#         if ann == 'NEITHER':
#             ann = 'NONE'
#
#         line = sig+" -> _"+ann+"_"
#         print(line)
#         lines.append(line)
# print("Number of annotations:", len(lines))
# with open(output_fname, 'w') as f:
#     f.write('\n'.join(lines))


# ### Script to find the number of flowdroid sources and sinks in the api
# from collections import Counter
# import re
# import pandas as pd
# from utils.FormattingTools import Preprocessor
#
#
# android_jar = 'Inputs/Android_30.csv'
# labels = ['NONE', 'SOURCE','SINK', 'INDSINK', 'IMPSOURCE']
# ss_fname = "Inputs/SourcesAndSinks_FlowDroid3.txt"#"Inputs/SourcesAndSinks_FlowDroid3.txt"#"Inputs/permissionMethodWithLabel.pscout"#x
#
# with open(ss_fname) as f:
#     lines = [l.rstrip() for l in f.readlines()]
#
# cntr = Counter()
# index, annotation = [],[]
# for line in lines:
#     line = line.strip()
#     mo = re.search(r"^ *(<.*?\(.*\)>)[^|]*_([^|]*?)_.*$", line)
#     if mo is not None:
#         # print(mo.group(0))
#         # print(mo.group(1))
#
#
#         # print(mo.group(2), "\n")
#         if len(mo.groups()) >= 2 and mo.group(2) in labels:
#             sig = mo.group(1)
#             sig = re.sub(r"^(<[^:]*: [^ ]+ [^( ]+) *\(", r"\1(", sig)
#             index.append(sig)
#             ann = mo.group(2)
#             if ann == 'INDSINK':
#                 ann = 'SINK'
#             if ann == 'IMPSOURCE':
#                 ann = 'SOURCE'
#             annotation.append(ann)
#         #print(len(mo.groups()))
#         cntr[mo.group(2)] += 1
#
#
# s = pd.Series(annotation, index=index)
#
#
# # pd.set_option('display.max_rows', 1000)
# pd.set_option('display.max_colwidth', 1000)
# # pd.set_option('display.min_rows', 125)
#
#
# s = s[~s.index.duplicated(keep='first')]
# print(s.value_counts())
# print(s.shape)
# s.index = s.index.astype(str)
# p = Preprocessor(docs_fname=android_jar)
# doc_df = p.preprocess_data()
# s_idx = p.stanardize_index(s)
# doc_idx = p.stanardize_index(doc_df)
# # print(s_idx)
# # print(doc_idx)
# print((s_idx.isin(doc_idx)).sum())
# print(s.shape[0])
# print((s_idx.isin(doc_idx)).sum()/s.shape[0])



# ### Script used to create manual feature cache
#
# import pandas as pd
# import os
# import json
#
# def extract_df_from_dir(dirname: str, filename: str = None):
#     '''
#     Convertes manual feature data stored in json files in 'dirname' to a single DataFrame and returns it. The
#     json files are expected to be the output from the ManualFeatureExtraction tool.
#     :param dirname:
#     :return:
#     '''
#
#     dataframes = []
#     for i, filename in enumerate(os.listdir(dirname)):
#         print("Loading " + filename + ".....")
#         if not os.path.isdir(dirname + '/' + filename):
#             with open(dirname + '/' + filename) as f:
#                 data = json.load(f)
#                 df = pd.DataFrame.from_dict(data, orient="index")
#                 # df = self.convert_str_to_float(df)
#                 print('df shape:', df.shape)
#             dataframes.append(df)
#     return pd.concat(dataframes)
# df = extract_df_from_dir('Inputs/DocOnlyFeats')
#
# print(df)
# # print((df=='NOT_SUPPORTED').sum())
# df.to_pickle("Inputs/Caches/DocOnlyManFeats.pickle")



# ##Script used for debugging. Makes sure the features are the same from the SuSi tool and our manual feature extractor
# prep = Preprocessor(man_feats_fname="Inputs/SuSiAnnFeats", use_man_feat_anns=True)
# lissa_feats = prep.preprocess_man_feats_file()
# susi_feats = pd.read_json("Inputs/SusiMethodFeats_fromSusi.json", orient="index")
# lissa_feats = lissa_feats.drop("Annotation", axis=1)
# lissa_feats = lissa_feats[sorted(lissa_feats.columns)]
# susi_feats = susi_feats[sorted(susi_feats.columns)]
# susi_feats = susi_feats.replace('TRUE', True)
# susi_feats = susi_feats.replace('FALSE', False)
#
#
# susi_feats = susi_feats.astype(bool)
# susi_feats = susi_feats.sort_index()
# lissa_feats = lissa_feats.replace(1.0, True)
# lissa_feats = lissa_feats.replace(0.0, False)
# lissa_feats = lissa_feats.astype(bool)
# lissa_feats.index = lissa_feats.index.str.replace(r"\)>.*", r")>", regex=True)
# susi_feats.index = prep.stanardize_index(susi_feats)
# susi_feats = susi_feats.sort_index()
# lissa_feats = lissa_feats.sort_index()
#
# print(lissa_feats.columns == susi_feats.columns)
# print('Lissa feats:', lissa_feats.shape)
# print('Susi feats:', susi_feats.shape)
# print("### Checking method signatures ###")
# print(lissa_feats.index[susi_feats.index != lissa_feats.index])
# print(susi_feats.index[susi_feats.index != lissa_feats.index])
#
# print("### Checking features ###")
# any_errors = False
# ok_cols = []
# pd.set_option('display.max_colwidth', 1000)
# print(susi_feats)
# print(lissa_feats)
# for i, col in enumerate(susi_feats.columns):
#     if not susi_feats[col].equals(lissa_feats[col]):
#         print("Error in col:", col)
#         # print(susi_feats[col].value_counts())
#         # print(lissa_feats[col].value_counts())
#         any_errors = True
#         # offending_meths = susi_feats.loc[susi_feats[col] !=lissa_feats[col], col].index.tolist()
#         # offending_meths.extend(lissa_feats.loc[susi_feats[col] !=lissa_feats[col], col].index.tolist())
#         error_df = pd.DataFrame()
#         error_df['Susi'] = susi_feats.loc[susi_feats[col] !=lissa_feats[col], col]
#         error_df['Lissa'] = lissa_feats.loc[susi_feats[col] !=lissa_feats[col], col]
#
#
#         # offending_meths = set(offending_meths)
#         print('\toffending methods:')
#         print(error_df)
#     else:
#         ok_cols.append(col)
#
#
# if not any_errors:
#     print("All clear")
# else:
#     print('Ok cols:', ok_cols)




