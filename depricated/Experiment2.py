import numpy as np
import pandas as pd
from utils.FormattingTools import Preprocessor
from Evaluation.Evaluator import Evaluator
from SSModel.cLiSSAfier import cLiSSAfier
import sys

'''
This experiments compares features on the NEW set of method annotations (api level 28 & 29) 
when trained on the OLD set of method annotations (api level 18 and below)
'''

cached_file = None#'Inputs/Caches/finalized_exp2_cache.pickle'
base_set_fname = "Inputs/FinalizedAnns_4_2_APIs.csv"
test_set_fname = "Inputs/FinalizedAnns_NewAPIs.csv"
docs_fname = "Inputs/android.csv"
man_feats_fname = "Inputs/Caches/DocOnlyManFeats.pickle"#"Inputs/manual_features.pickle"
fname_to_store_cache = 'Inputs/Caches/finalized_exp2_cache.pickle'
label_col_2_predict = 'Annotation'
model_type = 'SVM'
use_self_training = False
st_api_level = 18
'''
Manual Features Only (SuSi hyperparams)
Testing set:
              precision    recall  f1-score  support #Predictions
NEITHER        0.805594  0.775236  0.790123      743          715
SINK           0.375000  0.090909  0.146341       33            8
SOURCE         0.415162  0.513393  0.459082      224          277
macro avg      0.531919  0.459846  0.465182     1000         <NA>
weighted avg   0.703928  0.694000  0.694725     1000         <NA>
'''
'''
Return, Param, Signature, and Manual Features (SuSi hyperparams)
Testing set:
              precision    recall  f1-score  support #Predictions
NEITHER        0.836634  0.682369  0.751668      743          606
SINK           0.227273  0.151515  0.181818       33           22
SOURCE         0.400538  0.665179  0.500000      224          372
macro avg      0.488148  0.499687  0.477829     1000         <NA>
weighted avg   0.718839  0.661000  0.676489     1000         <NA>
'''
'''
Manual Features Only (SuSi hyperparams) ***********Both sets of manual features
Testing set:
              precision    recall  f1-score  support #Predictions
NEITHER        0.797030  0.866756  0.830432      743          808
SINK           0.428571  0.090909  0.150000       33            7
SOURCE         0.481081  0.397321  0.435208      224          185
macro avg      0.568894  0.451662  0.471880     1000         <NA>
weighted avg   0.714098  0.736000  0.719448     1000         <NA>
'''
'''
Return, Param, Signature, and Manual Features (SuSi hyperparams) ***********Both sets of manual features
Testing set:
              precision    recall  f1-score  support #Predictions
NEITHER        0.818182  0.678331  0.741722      743          616
SINK           0.285714  0.242424  0.262295       33           28
SOURCE         0.356742  0.566964  0.437931      224          356
macro avg      0.486879  0.495907  0.480649     1000         <NA>
weighted avg   0.697248  0.639000  0.657852     1000         <NA>
'''
'''
Manual Features Only
Testing set:
              precision    recall  f1-score  support #Predictions
NEITHER        0.774752  0.421265  0.545772      743          404
SINK           0.071429  0.484848  0.124514       33          224
SOURCE         0.376344  0.625000  0.469799      224          372
macro avg      0.407508  0.510371  0.380028     1000         <NA>
weighted avg   0.662299  0.469000  0.514852     1000         <NA>
'''
'''
Return, Param, Signature, and Manual Features
              precision    recall  f1-score  support #Predictions
NEITHER        0.854348  0.528937  0.653367      743          460
SINK           0.200000  0.272727  0.230769       33           45
SOURCE         0.353535  0.781250  0.486787      224          495
macro avg      0.469294  0.527638  0.456974     1000         <NA>
weighted avg   0.720572  0.577000  0.602107     1000         <NA>
'''
'''
Manual Features Only ***********Both sets of manual features
Testing set:
              precision    recall  f1-score  support #Predictions
NEITHER        0.787190  0.512786  0.621027      743          484
SINK           0.100840  0.363636  0.157895       33          119
SOURCE         0.345088  0.611607  0.441224      224          397
macro avg      0.411040  0.496010  0.406715     1000         <NA>
weighted avg   0.665510  0.530000  0.565468     1000         <NA>
'''
'''
Return, Param, Signature, and Manual Features ***********Both sets of manual features
              precision    recall  f1-score  support #Predictions
NEITHER        0.834933  0.585464  0.688291      743          521
SINK           0.312500  0.303030  0.307692       33           32
SOURCE         0.340045  0.678571  0.453055      224          447
macro avg      0.495826  0.522355  0.483013     1000         <NA>
weighted avg   0.706838  0.597000  0.623039     1000         <NA>
'''
'''
Return, Param, and Signature Features
              precision    recall  f1-score  support #Predictions
NEITHER        0.868132  0.531629  0.659432      743          455
SINK           0.163265  0.242424  0.195122       33           49
SOURCE         0.375000  0.830357  0.516667      224          496
macro avg      0.468799  0.534803  0.457074     1000         <NA>
weighted avg   0.734410  0.589000  0.612131     1000         <NA>
'''

#Experimental parameters
ex_params = {
    'Manual_Feats': True,

    # # Method descriptions column and hyperparameters
    # 'Description': {'min_df' : 3,  'max_df' : .8, 'max_features' : 20000, 'ngram_range': (1, 4)},

    # # Return value descriptions column
    # 'Return': {'min_df' : 3,  'max_df' : .8, 'max_features' : 2000, 'ngram_range': (1, 1)},
    #
    # # Method parameter values descriptions column
    # 'Parameters': {'min_df' : 3,  'max_df' : .8, 'max_features' : 4000, 'ngram_range': (1, 3)},
    #
    # # Method signature column
    # 'SigFeatures': {'min_df' : 3,  'max_df' : .8, 'max_features' : 8000, 'ngram_range': (1, 3)},
}
ex_params_tilde = {
    'Manual_Feats': True,

    # Method descriptions column and hyperparameters
    'Description': {'min_df' : 3,  'max_df' : .8, 'max_features' : 20000, 'ngram_range': (1, 4)},

    # Return value descriptions column
    'Return': {'min_df' : 3,  'max_df' : .8, 'max_features' : 2000, 'ngram_range': (1, 1)},

    # Method parameter values descriptions column
    'Parameters': {'min_df' : 3,  'max_df' : .8, 'max_features' : 2000, 'ngram_range': (1, 3)},

    # Method signature column
    'SigFeatures': {'min_df' : 3,  'max_df' : .8, 'max_features' : 8000, 'ngram_range': (1, 3)},
    }

def print_stats(df=None, name=''):
    print("From "+name+" annotations:")
    print("Percent of sources:"
          , df.loc[ (df[label_col_2_predict] =='SOURCE')].shape[0]/df.loc[ df[label_col_2_predict].notna()].shape[0])
    print("Percent of sinks:"
          ,df.loc[ (df[label_col_2_predict] =='SINK')].shape[0]/df.loc[ df[label_col_2_predict].notna()].shape[0])
    print("Percent of neither:"
          ,df.loc[ (df[label_col_2_predict] =='NEITHER')].shape[0]/df.loc[ df[label_col_2_predict].notna()].shape[0])


#Preprocessing step (this is customized for our files)
if cached_file is None:
    preprocessor = Preprocessor(anns_fname=base_set_fname, docs_fname=docs_fname, man_feats_fname=man_feats_fname)
    df = preprocessor.preprocess_data()
    if fname_to_store_cache is not None:
        df.to_pickle(fname_to_store_cache)
else:
    df = pd.read_pickle(cached_file)

print_stats(df, base_set_fname)


classifier = cLiSSAfier(model_type)

df['is_train_set'] = True
if use_self_training:
    preprocessor = Preprocessor(docs_fname=docs_fname, man_feats_fname=man_feats_fname
                                , up_to_api_level=st_api_level)
    df_tilde = preprocessor.preprocess_data()
else:
    df_tilde, ex_params_tilde = None, None

preprocessor = Preprocessor(anns_fname=test_set_fname, docs_fname=docs_fname, man_feats_fname=man_feats_fname)
test_set = preprocessor.preprocess_data()
test_set['is_train_set'] = False
print_stats(test_set, test_set_fname)
df = df.append(test_set)
ev = Evaluator(X=df, y=df[label_col_2_predict], model=classifier, feat_cols_and_hyperparams=ex_params, X_tilde=df_tilde
               , feat_cols_and_hyperparams_tilde=ex_params_tilde)
ev.classic_eval(split_on_col='is_train_set')

# if model_type == 'SVM':
#     analysis_df = ev.get_feature_analysis_dataframe() #Columns: NEITHER, SINK, SOURCE
#     print(analysis_df['SOURCE'].sort_values(ascending=False))