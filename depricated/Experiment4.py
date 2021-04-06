import numpy as np
import pandas as pd
from utils.FormattingTools import Preprocessor
from Evaluation.Evaluator import Evaluator
from SSModel.cLiSSAfier import cLiSSAfier
from SSModel.InternalSSVectorizers.BoWVectorizer import BoWVectorizer
import sys
'''
This experiments compares features on the NEW set of method annotations (api level 28 & 29) only
10 fold cross-validation
'''


cached_file = None#'Inputs/Caches/finalized_exp4_cache.pickle'
annotation_set_fname = "Inputs/FinalizedAnns_NewAPIs.csv"
docs_fname = "Inputs/android.csv"
man_feats_fname = "Inputs/Caches/DocOnlyManFeats.pickle"#"Inputs/doc_man_feats.pickle"#"Inputs/manual_features.pickle"
fname_to_store_cache = 'Inputs/Caches/finalized_exp4_cache.pickle'
label_col_2_predict = 'Annotation'
remove_no_man_feats = True
model_type = 'SVM'
num_folds = 10
runs_to_average = 5
'''
SuSi tool:
Class details: === Detailed Accuracy By Class ===

               TP Rate   FP Rate   Precision   Recall  F-Measure   ROC Area  Class
                 0.219     0.023      0.731     0.219     0.337      0.63     source
                 0.091     0.003      0.5       0.091     0.154      0.731    sink
                 0.974     0.79       0.781     0.974     0.867      0.592    neithernor
Weighted Avg.    0.776     0.592      0.761     0.776     0.725      0.605
'''
'''
Manual Features Only (SuSi hyperparams)
!!!!!! After averaging 5 runs, final tally:
              precision    recall  f1-score     support #Predictions
NEITHER        0.780811  0.973901  0.865992   74.300000        92.68
SINK           0.053333  0.011619  0.019012    3.446667     0.335556
SOURCE         0.768080  0.236262  0.350094   22.400000            7
macro avg      0.542419  0.415436  0.419828  100.000000         <NA>
weighted avg   0.759249  0.777000  0.724263  100.000000         <NA>
'''
'''
Return, Param, Signature, and Manual Features (SuSi hyperparams)
              precision    recall  f1-score  support #Predictions
NEITHER        0.892745  0.906393  0.898639     74.3        75.48
SINK           0.326667  0.210111  0.236947      3.3         2.12
SOURCE         0.737511  0.731771  0.729776     22.4         22.4
macro avg      0.652307  0.616092  0.621788    100.0         <NA>
weighted avg   0.843947  0.845200  0.841243    100.0         <NA>

'''
'''
Man Feats only
!!!!!! After averaging 5 runs, final tally: 
               precision    recall  f1-score
Source          0.517910  0.540214  0.521399
Sink            0.306214  0.390952  0.315378
Neither         0.834997  0.813364  0.822915
Macro Avg.      0.553040  0.581510  0.553231
Weighted Avg.   0.751237  0.738200  0.741285
'''
'''
Return, Param, Signature, and Manual Features
!!!!!! After averaging 5 runs, final tally:
              precision    recall  f1-score     support #Predictions
NEITHER        0.915659  0.870809  0.891810   74.300000         70.7
SINK           0.384143  0.327937  0.326481    3.373333      2.83778
SOURCE         0.694598  0.820400  0.747700   22.400000        26.52
macro avg      0.661784  0.670863  0.653006  100.000000         <NA>
weighted avg   0.852586  0.841800  0.842971  100.000000         <NA>
'''


'''
Man Feats only  ***********Both sets of manual features
!!!!!! After averaging 5 runs, final tally:
              precision    recall  f1-score  support #Predictions
NEITHER        0.893838  0.757414  0.818775     74.3        62.96
SINK           0.154361  0.574667  0.230930      3.3        12.78
SOURCE         0.666588  0.724499  0.688973     22.4        24.26
macro avg      0.571596  0.685527  0.579559    100.0         <NA>
weighted avg   0.822619  0.743000  0.772550    100.0         <NA>
'''
'''
Return, Param, Signature, and Manual Features  ***********Both sets of manual features
!!!!!! After averaging 5 runs, final tally:
              precision    recall  f1-score     support #Predictions
NEITHER        0.912842  0.880735  0.896002   74.300000        71.66
SINK           0.340407  0.275222  0.280453    3.373333      2.73778
SOURCE         0.711696  0.817323  0.757480   22.400000        25.66
macro avg      0.652957  0.655908  0.642806  100.000000         <NA>
weighted avg   0.852955  0.846200  0.846529  100.000000         <NA>

'''
'''
Return, Param, and Signature Features
!!!!!! After averaging 5 runs, final tally:
              precision    recall  f1-score     support #Predictions
NEITHER        0.912791  0.869462  0.889936   74.300000        70.78
SINK           0.340519  0.234762  0.250891    3.373333      2.43556
SOURCE         0.691096  0.826010  0.748183   22.400000        26.84
macro avg      0.645148  0.641189  0.627166  100.000000         <NA>
weighted avg   0.848306  0.838800  0.838992  100.000000         <NA>
'''


#Experimental parameters for SVM
ex_params = {
    'Manual_Feats': True,

    # # Method descriptions column and hyperparameters
    # 'Description': {'min_df' : 3,  'max_df' : .8, 'max_features' : 20000, 'ngram_range': (1, 4)},
    #
    # # Return value descriptions column
    # 'Return': {'min_df' : 3,  'max_df' : .8, 'max_features' : 2000, 'ngram_range': (1, 1)},
    # # # #
    # # Method parameter values descriptions column
    # 'Parameters': {'min_df' : 3,  'max_df' : .8, 'max_features' : 5000, 'ngram_range': (1, 1)},
    # # # #
    # # Method signature column
    # 'SigFeatures': {'min_df' : 3,  'max_df' : .8, 'max_features' : 8000, 'ngram_range': (1, 1)},
}

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
    pd.set_option('display.max_rows', 50)

    print(analysis_df['SINK'].sort_values(ascending=False))
    print((analysis_df.sum(axis=1).abs() < 0.000001).sum())