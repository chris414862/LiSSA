import numpy as np
import pandas as pd
from utils.FormattingTools import Preprocessor
from Evaluation.Evaluator import Evaluator
from SSModel.cLiSSAfier import cLiSSAfier
from SSModel.InternalSSVectorizers.BoWVectorizer import BoWVectorizer
import sys

'''
This experiments compares features on the OLD set of method annotations (api level 18 and below) only
10 fold cross-validation
'''

cached_file = None#'Inputs/Caches/finalized_exp1_cache.pickle'
annotation_set_fname = "Inputs/FinalizedAnns_4_2_APIs.csv"
docs_fname = "Inputs/android.csv"
man_feats_fname = "Inputs/Caches/DocOnlyManFeats.pickle"#"Inputs/manual_features.pickle""Inputs/SuSiAnnFeats"#
fname_to_store_cache = 'Inputs/Caches/finalized_exp1_cache.pickle'#'Inputs/Caches/susi_anns_cache.pickle'
label_col_2_predict = 'Annotation'
remove_no_man_feats = True
model_type = 'SVM'
num_folds = 10
runs_to_average = 5

#Experimental parameters for SVM
ex_params = {
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
'''
Class details: === Detailed Accuracy By Class ===

               TP Rate   FP Rate   Precision   Recall  F-Measure   ROC Area  Class
                 0.514     0.048      0.643     0.514     0.571      0.851    source
                 0.325     0.016      0.741     0.325     0.452      0.799    sink
                 0.925     0.573      0.816     0.925     0.867      0.676    neithernor
Weighted Avg.    0.792     0.429      0.782     0.792     0.773      0.716

'''
'''
Manual Features Only (SuSi hyperparams)
!!!!!! After averaging 5 runs, final tally:
              precision    recall  f1-score  support #Predictions
NEITHER        0.831085  0.917990  0.871536     73.3        80.96
SINK           0.738951  0.334088  0.443683     12.3         5.52
SOURCE         0.668226  0.630020  0.639016     14.4        13.52
macro avg      0.746087  0.627366  0.651412    100.0         <NA>
weighted avg   0.801228  0.802600  0.786086    100.0         <NA>

'''
'''
Return, Param, Signature, and Manual Features (SuSi hyperparams)
!!!!!! After averaging 5 runs, final tally:
              precision    recall  f1-score  support #Predictions
NEITHER        0.876502  0.915147  0.894636     73.3        76.56
SINK           0.689948  0.547805  0.599991     12.3         9.78
SOURCE         0.778237  0.732209  0.746070     14.4        13.66
macro avg      0.781562  0.731721  0.746899    100.0         <NA>
weighted avg   0.841932  0.842000  0.837223    100.0         <NA>
'''

'''
Manual Features Only (SuSi hyperparams) ***********Both sets of manual features
!!!!!! After averaging 5 runs, final tally:
              precision    recall  f1-score  support #Predictions
NEITHER        0.833225  0.914986  0.871353     73.3         80.5
SINK           0.740667  0.332273  0.442116     12.3         5.56
SOURCE         0.671426  0.645029  0.648529     14.4        13.94
macro avg      0.748439  0.630763  0.654000    100.0         <NA>
weighted avg   0.801914  0.803200  0.787213    100.0         <NA>
'''
'''
Return, Param, Signature, and Manual Features (SuSi hyperparams) ***********Both sets of manual features
!!!!!! After averaging 5 runs, final tally:
              precision    recall  f1-score  support #Predictions
NEITHER        0.871180  0.913929  0.891541     73.3        76.88
SINK           0.693765  0.526431  0.587654     12.3         9.42
SOURCE         0.744042  0.716902  0.723959     14.4         13.7
macro avg      0.769663  0.719087  0.734385    100.0         <NA>
weighted avg   0.834611  0.836600  0.831236    100.0         <NA>
'''
'''
Return, Param, and Signature Features (SuSi Features)
!!!!!! After averaging 5 runs, final tally:
              precision    recall  f1-score  support #Predictions
NEITHER        0.869357  0.927275  0.896927     73.3         78.2
SINK           0.708207  0.501680  0.577267     12.3         8.58
SOURCE         0.763222  0.708867  0.727013     14.4        13.22
macro avg      0.780262  0.712607  0.733736    100.0         <NA>
weighted avg   0.840127  0.843200  0.835571    100.0         <NA>
'''
'''
Manual Features Only 

'''
'''
Return, Param, Signature, and Manual Features
!!!!!! After averaging 5 runs, final tally:
              precision    recall  f1-score  support #Predictions
NEITHER        0.906600  0.855103  0.879099     73.3        69.16
SINK           0.603922  0.658716  0.615798     12.3        13.48
SOURCE         0.683577  0.833679  0.743816     14.4        17.36
macro avg      0.731366  0.782499  0.746238    100.0         <NA>
weighted avg   0.842676  0.825400  0.829403    100.0         <NA>
'''
'''
Manual Features Only  ***********Both sets of manual features
!!!!!! After averaging 5 runs, final tally:
              precision    recall  f1-score  support #Predictions
NEITHER        0.886634  0.539966  0.665676     73.3         44.7
SINK           0.294200  0.697658  0.403671     12.3        30.02
SOURCE         0.500563  0.885044  0.634750     14.4        25.28
macro avg      0.560465  0.707556  0.568032    100.0         <NA>
weighted avg   0.763549  0.608600  0.633285    100.0         <NA>
'''
'''
Return, Param, Signature, and Manual Features  ***********Both sets of manual features
!!!!!! After averaging 5 runs, final tally:
              precision    recall  f1-score  support #Predictions
NEITHER        0.905978  0.855872  0.879172     73.3        69.26
SINK           0.593048  0.655235  0.610570     12.3        13.44
SOURCE         0.685519  0.822062  0.740711     14.4         17.3
macro avg      0.728182  0.777723  0.743484    100.0         <NA>
weighted avg   0.841640  0.825400  0.829108    100.0         <NA>
'''
'''
Return, Param, and Signature Features
!!!!!! After averaging 5 runs, final tally:
              precision    recall  f1-score  support #Predictions
NEITHER        0.898921  0.857551  0.877038     73.3        69.94
SINK           0.582286  0.597227  0.581162     12.3        12.74
SOURCE         0.687838  0.829791  0.745472     14.4        17.32
macro avg      0.723015  0.761523  0.734558    100.0         <NA>
weighted avg   0.833605  0.821200  0.824027    100.0         <NA>
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
    preprocessor = Preprocessor(anns_fname=annotation_set_fname, docs_fname=docs_fname, man_feats_fname=man_feats_fname)
                                #, remove_no_man_feats=remove_no_man_feats)
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
               , runs_to_average=runs_to_average)
ev.kfold_eval(folds=num_folds)

if model_type == 'SVM':
    analysis_df = ev.get_feature_analysis_dataframe() #Columns: NEITHER, SINK, SOURCE
    pd.set_option('display.max_colwidth', 1000)
    pd.set_option('display.min_rows', 50)
    print(analysis_df[['SINK','SOURCE']].abs().sum(axis=1).sort_values(ascending=False))
    print((analysis_df[['SINK','SOURCE']].abs().sum(axis=1) < 0.000001).sum())
    null_svm_weights= analysis_df[['SINK','SOURCE']].index[(analysis_df.abs().sum(axis=1) < 0.000001)].tolist()
    print(null_svm_weights)

    feat_cols = df[mf_col].sum(axis=0)
    print(feat_cols.index[feat_cols <= 1.0])
    feat_cols[feat_cols <= 1.0].to_csv("in_feat_domain_no_feats.csv")












