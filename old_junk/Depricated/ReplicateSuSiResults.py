import pandas as pd
import re
from collections import Counter

#### Prep SuSi's Annotated methods for our feature extraction tool
# input_fname= "Inputs/permissionMethodWithLabel.pscout"
# output_fname='Inputs/susi_annotations.csv'
# labels = ['NONE', 'SOURCE','SINK', 'INDSINK', 'IMPSOURCE']
#
# with open(input_fname, encoding='utf-8') as f:
#
#     lines = [line.rstrip() for line in f.readlines()]
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
# s.to_csv(output_fname)

####################################################################################################
####################################################################################################

import pandas as pd
from utils.FormattingTools import Preprocessor
from Evaluation.Evaluator import Evaluator
from SSModel.cLiSSAfier import cLiSSAfier
from SSModel.InternalSSVectorizers.BoWVectorizer import BoWVectorizer



##### Get features from our manual feature extractor (written to file) and classify
cached_file = 'Inputs/Caches/SuSiReplication_cache.pickle'
annotation_set_fname = None
use_man_feat_anns = True #Use the annotation set that was expanded in the manual feature extraction tool
docs_fname = "Inputs/android.csv"
man_feats_fname = "Inputs/SuSiAnnFeats"#"Inputs/manual_features.pickle""Inputs/SuSiAnnFeats"#
fname_to_store_cache = 'Inputs/Caches/SuSiReplication_cache.pickle'#'Inputs/Caches/susi_anns_cache.pickle'
label_col_2_predict = 'Annotation'

model_type = 'SVM'
num_folds = 10
runs_to_average = 5

#Experimental parameters for SVM
ex_params = {
    'Manual_Feats': True,

    # # Method descriptions column and hyperparameters
    # 'Description': {'min_df' : 3,  'max_df' : .8, 'max_features' : 20000, 'ngram_range': (1, 4)},
    # #
    # # Return value descriptions column
    # 'Return': {'min_df' : 3,  'max_df' : .8, 'max_features' : 2000, 'ngram_range': (1, 1)},
    # # # # #
    # # Method parameter values descriptions column
    # 'Parameters': {'min_df' : 3,  'max_df' : .8, 'max_features' : 5000, 'ngram_range': (1, 3)},
    # # # #
    # # Method signature column
    # 'SigFeatures': {'min_df' : 3,  'max_df' : .8, 'max_features' : 8000, 'ngram_range': (1, 3)},
}

#### From SuSi Tool:
'''
Run 1:
Class details: === Detailed Accuracy By Class ===

               TP Rate   FP Rate   Precision   Recall  F-Measure   ROC Area  Class
                 0.872     0.025      0.872     0.872     0.872      0.962    source
                 0.781     0.044      0.855     0.781     0.816      0.893    sink
                 0.924     0.159      0.892     0.924     0.908      0.883    neithernor
Weighted Avg.    0.88      0.108      0.88      0.88      0.879      0.899
Run 2:
Class details: === Detailed Accuracy By Class ===

               TP Rate   FP Rate   Precision   Recall  F-Measure   ROC Area  Class
                 0.86      0.028      0.86      0.86      0.86       0.958    source
                 0.789     0.051      0.835     0.789     0.811      0.892    sink
                 0.911     0.159      0.891     0.911     0.901      0.877    neithernor
Weighted Avg.    0.873     0.111      0.872     0.873     0.872      0.894

'''

#### From Lissa:
'''
!!!!!! After averaging 5 runs, final tally:
              precision    recall  f1-score  support #Predictions
NEITHER        0.865517  0.913632  0.887914     29.2        30.82
SINK           0.836157  0.794684  0.809788     13.4        12.78
SOURCE         0.896432  0.799346  0.836504      9.2          8.2
macro avg      0.866036  0.835888  0.844735     51.8         <NA>
weighted avg   0.868592  0.862926  0.861706     51.8         <NA>
'''

def print_stats(df=None):
    print("From annotations:")
    print("Percent of sources:"
          , df.loc[ (df[label_col_2_predict] =='SOURCE')].shape[0]/df.loc[ df[label_col_2_predict].notna()].shape[0])
    print("Percent of sinks:"
          ,df.loc[ (df[label_col_2_predict] =='SINK')].shape[0]/df.loc[ df[label_col_2_predict].notna()].shape[0])
    print("Percent of neither:"
          ,df.loc[ (df[label_col_2_predict] =='NEITHER')].shape[0]/df.loc[ df[label_col_2_predict].notna()].shape[0])


#Preprocessing step (this is customized for our files)
if cached_file is None:
    preprocessor = Preprocessor(anns_fname=annotation_set_fname, docs_fname=docs_fname, man_feats_fname=man_feats_fname
                                , use_man_feat_anns=use_man_feat_anns)
    df = preprocessor.preprocess_data()
    if fname_to_store_cache is not None:
        df.to_pickle(fname_to_store_cache)
else:
    df = pd.read_pickle(cached_file)

print_stats(df)
bow = BoWVectorizer()
mf_col = bow.find_man_feat_cols(df)
print(len(mf_col))
print(df[mf_col].sum(axis=0).value_counts().sort_index())
col_cnts = df[mf_col].sum(axis=0)
print(col_cnts.loc[col_cnts <= 0.0].shape)

classifier = cLiSSAfier(model_type)

ev = Evaluator(X=df, y=df[label_col_2_predict], model=classifier, feat_cols_and_hyperparams=ex_params
               , runs_to_average=runs_to_average, use_man_feat_anns=use_man_feat_anns)
ev.kfold_eval(folds=num_folds)

if model_type == 'SVM':
    analysis_df = ev.get_feature_analysis_dataframe() #Columns: NEITHER, SINK, SOURCE
    print(analysis_df.columns)
    pd.set_option('display.min_rows', 100)
    print(analysis_df[['SINK']].abs().sum(axis=1).sort_values(ascending=False).iloc[:50])
    analysis_df[['SINK']].abs().sum(axis=1).sort_values(ascending=False).iloc[:50].to_csv("top_susi_weights.csv")
    print((analysis_df[['SINK']].abs().sum(axis=1) < 0.000001).sum())
