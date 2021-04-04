import numpy as np
import pandas as pd
from utils.FormattingTools import Preprocessor
from Evaluation.Evaluator import Evaluator
from SSModel.cLiSSAfier import cLiSSAfier
import sys
'''
This experiment compares features on the NEW set of method annotations (api level 28 & 29) 
but adds the OLD annotations to each fold. Simulates an update of annotations for new API version
10 fold cross-validation
'''


cached_file = None#'Inputs/Caches/finalized_exp4_cache.pickle'
annotation_set_fname = "Inputs/FinalizedAnns_NewAPIs.csv"
add_to_train_fold_ann_fname = "Inputs/FinalizedAnns_4_2_APIs.csv"
docs_fname = "Inputs/android.csv"
man_feats_fname = "Inputs/Caches/DocOnlyManFeats.pickle"#"Inputs/doc_man_feats.pickle"#"Inputs/manual_features.pickle"
fname_to_store_cache = 'Inputs/Caches/finalized_exp4_cache.pickle'
label_col_2_predict = 'Annotation'
remove_no_man_feats = True
model_type = 'SVM'
num_folds = 10
runs_to_average = 5

#Experimental parameters for SVM
ex_params = {
    'Manual_Feats': True,

    # # Method descriptions column and hyperparameters
    # 'Description': {'min_df' : 3,  'max_df' : .8, 'max_features' : 20000, 'ngram_range': (1, 3)},
    #
    # Return value descriptions column
    'Return': {'min_df' : 3,  'max_df' : .8, 'max_features' : 2000, 'ngram_range': (1, 3)},
    # # # #
    # Method parameter values descriptions column
    'Parameters': {'min_df' : 3,  'max_df' : .8, 'max_features' : 5000, 'ngram_range': (1, 3)},
    # # #
    # Method signature column
    'SigFeatures': {'min_df' : 3,  'max_df' : .8, 'max_features' : 8000, 'ngram_range': (1, 3)},
}

'''
Manual Features Only (SuSi hyperparams)
!!!!!! After averaging 5 runs, final tally:
              precision    recall  f1-score     support #Predictions
NEITHER        0.786829  0.959646  0.863867   74.300000        90.66
SINK           0.268333  0.121339  0.154698    3.373333      1.01778
SOURCE         0.713602  0.269241  0.381291   22.400000         8.34
macro avg      0.593543  0.453989  0.470603  100.000000         <NA>
weighted avg   0.758794  0.776200  0.732944  100.000000         <NA>
'''
'''
Return, Param, Signature, and Manual Features (SuSi hyperparams)
!!!!!! After averaging 5 runs, final tally:
              precision    recall  f1-score     support #Predictions
NEITHER        0.892235  0.930279  0.910271   74.300000        77.48
SINK           0.426815  0.289397  0.307035    3.373333         1.98
SOURCE         0.790919  0.730180  0.754851   22.400000        20.58
macro avg      0.707062  0.654156  0.661512  100.000000         <NA>
weighted avg   0.859539  0.863800  0.857572  100.000000         <NA>
'''

'''
Manual Features Only (SuSi hyperparams) ***********Both sets of manual features

'''
'''
Manual Features Only
!!!!!! After averaging 5 runs, final tally:
               precision    recall  f1-score
Source          0.486802  0.533441  0.503295
Sink            0.316667  0.280905  0.256524
Neither         0.823276  0.808222  0.814593
Macro Avg.      0.542248  0.540856  0.524804
Weighted Avg.   0.736550  0.728600  0.728790
'''
'''
Return, Param, Signature, and Manual Features
!!!!!! After averaging 5 runs, final tally:
              precision    recall  f1-score     support #Predictions
NEITHER        0.926458  0.850060  0.885926   74.300000        68.16
SINK           0.429074  0.466540  0.401986    3.373333      3.37111
SOURCE         0.670668  0.851952  0.747656   22.400000        28.54
macro avg      0.671820  0.719464  0.675153  100.000000         <NA>
weighted avg   0.856835  0.836400  0.840674  100.000000         <NA>
'''

'''
Manual Features Only  ***********Both sets of manual features
!!!!!! After averaging 5 runs, final tally:
              precision    recall  f1-score  support #Predictions
NEITHER        0.898898  0.564009  0.690738     74.3        46.62
SINK           0.103461  0.513571  0.164480      3.3         16.4
SOURCE         0.508653  0.829293  0.626363     22.4        36.98
macro avg      0.503671  0.635624  0.493860    100.0         <NA>
weighted avg   0.787983  0.621400  0.661182    100.0         <NA>
'''
'''
Return, Param, Signature, and Manual Features  ***********Both sets of manual features
!!!!!! After averaging 5 runs, final tally:
              precision    recall  f1-score  support #Predictions
NEITHER        0.921647  0.851800  0.884625     74.3        68.68
SINK           0.413524  0.408857  0.386556      3.3         3.32
SOURCE         0.671697  0.840963  0.741988     22.4           28
macro avg      0.668956  0.700540  0.671056    100.0         <NA>
weighted avg   0.853652  0.834400  0.838767    100.0         <NA>
'''
'''
Return, Param, and Signature Features
!!!!!! After averaging 5 runs, final tally:
              precision    recall  f1-score     support #Predictions
NEITHER        0.924652  0.844465  0.882126   74.300000        67.88
SINK           0.335466  0.361989  0.323203    3.446667      3.63111
SOURCE         0.670056  0.854075  0.747963   22.400000        28.64
macro avg      0.639220  0.682847  0.647297  100.000000         <NA>
weighted avg   0.852484  0.831200  0.836307  100.000000         <NA>
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

if add_to_train_fold_ann_fname is not None:
    preprocessor = Preprocessor(anns_fname=add_to_train_fold_ann_fname, docs_fname=docs_fname
                                , man_feats_fname=man_feats_fname)
    # , remove_no_man_feats=remove_no_man_feats)
    add_to_train_fold_df = preprocessor.preprocess_data()
else:
    add_to_train_fold_df = None

classifier = cLiSSAfier(model_type)

ev = Evaluator(X=df, y=df[label_col_2_predict], model=classifier, feat_cols_and_hyperparams=ex_params
               , add_to_train_fold_X=add_to_train_fold_df, add_to_train_fold_y=add_to_train_fold_df[label_col_2_predict]
               , runs_to_average=runs_to_average)
ev.kfold_eval(folds=num_folds)

if model_type == 'SVM':
    analysis_df = ev.get_feature_analysis_dataframe() #Columns: NEITHER, SINK, SOURCE
    print(analysis_df)
