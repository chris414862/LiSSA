from sklearn.metrics import classification_report
import pandas as pd
from utils.FormattingTools import Preprocessor
from SSModel.cLiSSAfier import cLiSSAfier
import sys

annotation_set1_fname = "Inputs/FinalizedAnns_4_2_APIs.csv"
annotation_set2_fname = "Inputs/FinalizedAnns_NewAPIs.csv"
docs_fname = "Inputs/android_30.csv"
man_feats_fname = "Inputs/Caches/DocOnlyManFeats.pickle"#"Inputs/manual_features.pickle""Inputs/SuSiAnnFeats"#
model_type = 'SVM'
cache_name = "Inputs/Caches/SSPreprocess_cache.pickle"

hyperparams = {
    'Manual_Feats': True,

    # # Method descriptions column and hyperparameters
    # 'Description': {'min_df' : 3,  'max_df' : .8, 'max_features' : 20000, 'ngram_range': (1, 4)},
    # # #
    # Return value descriptions column
    'Return': {'min_df' : 3,  'max_df' : .8, 'max_features' : 2000, 'ngram_range': (1, 1)},
    # # # #
    # Method parameter values descriptions column
    'Parameters': {'min_df' : 3,  'max_df' : .8, 'max_features' : 5000, 'ngram_range': (1, 3)},
    # # #
    # Method signature column
    'SigFeatures': {'min_df' : 3,  'max_df' : .8, 'max_features' : 8000, 'ngram_range': (1, 3)},
}
if cache_name is None:
    preprocessor = Preprocessor(anns_fname=annotation_set1_fname, docs_fname=docs_fname, man_feats_fname=man_feats_fname)
    df1 = preprocessor.preprocess_data()

    preprocessor = Preprocessor(anns_fname=annotation_set2_fname, docs_fname=docs_fname,
                                man_feats_fname=man_feats_fname)
    df2 = preprocessor.preprocess_data()
    df = df1.append(df2)
    preprocessor = Preprocessor(docs_fname=docs_fname,
                                man_feats_fname=man_feats_fname)
    df3 = preprocessor.preprocess_data()

    df['train'] = True
    df3['train'] = False # Inference set
    df = df.append(df3)
    df.to_pickle("Inputs/Caches/SSPreprocess_cache.pickle")
else:
    df = pd.read_pickle(cache_name)

pd.set_option('display.max_colwidth', 1000)
print(df.columns)
print("check getDeviceId:")
print( df.loc[df["MethodName"]=="getDeviceId",['Annotation','MethodName']])

df_train = df.loc[df['train']==True]
df_inference = df.loc[df['train']==False]

classifier = cLiSSAfier(model_type)
classifier.train(df_train, df_train['Annotation'], feat_cols_and_hyperparams=hyperparams)

preds_train, _ = classifier.predict(df_train)
preds_inference, dec_func = classifier.predict(df_inference)
# preds, dec_func = classifier.predict(df_inference)
col0 = dec_func[0]
col1:pd.Series = dec_func[1]
col2:pd.Series = dec_func[2]
by_dist:pd.Series = dec_func[1].sort_values(ascending=False).iloc[:30]

print(classifier.internal_model.get_internal_model().classes_)

srcs = preds_inference[preds_inference == 'SOURCE']
by_dist:pd.Series = dec_func[2]
srcs_by_dist:pd.Series = by_dist[srcs.index].sort_values(ascending=False)

srcs_by_dist.name = "HypPlaneDist"
print(">>>")
print(srcs_by_dist[:30])
srcs_by_dist.to_csv("SourcesByConfidence.csv")



sinks = preds_inference[preds_inference == 'SINK']
by_dist:pd.Series = dec_func[1]
sinks_by_dist:pd.Series = by_dist[sinks.index].sort_values(ascending=False)

sinks_by_dist.name = "HypPlaneDist"
print(">>>")
print(sinks_by_dist[:30])
sinks_by_dist.to_csv("SinksByConfidence.csv")
# print(preds.shape)
# print(classification_report(df_train['Annotation'], preds_train))

