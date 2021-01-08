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

cached_file = 'Inputs/Caches/finalized_exp6_cache.pickle'
foundation_set_fname = "Inputs/FinalizedAnns_4_2_APIs.csv"
base_set_fname = "Inputs/FinalizedAnns_NewAPIs.csv"

docs_fname = "Inputs/android.csv"
man_feats_fname = "Inputs/Caches/DocOnlyManFeats.pickle"#"Inputs/manual_features.pickle"
fname_to_store_cache = 'Inputs/Caches/finalized_exp6_cache.pickle'
label_col_2_predict = 'Annotation'
model_type = 'SVM'
st_api_level = 18
level_to_start_test = 29


#Experimental parameters
ex_params = {
    'Manual_Feats': True,

    # # Method descriptions column and hyperparameters
    # 'Description': {'min_df' : 3,  'max_df' : .8, 'max_features' : 20000, 'ngram_range': (1, 4)},

    # Return value descriptions column
    'Return': {'min_df' : 3,  'max_df' : .8, 'max_features' : 2000, 'ngram_range': (1, 1)},

    # Method parameter values descriptions column
    'Parameters': {'min_df' : 3,  'max_df' : .8, 'max_features' : 4000, 'ngram_range': (1, 3)},

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
    if foundation_set_fname is not None:
        preprocessor = Preprocessor(anns_fname=foundation_set_fname, docs_fname=docs_fname, man_feats_fname=man_feats_fname)
        df2 = preprocessor.preprocess_data()
        df = df.append(df2)
    if fname_to_store_cache is not None:
        df.to_pickle(fname_to_store_cache)
else:
    df = pd.read_pickle(cached_file)

print_stats(df, base_set_fname)
classifier = cLiSSAfier(model_type)



df['is_train_set'] = True
print(df['ApiLevel'].value_counts().sort_index().cumsum())
df.loc[df['ApiLevel'] >= level_to_start_test, 'is_train_set'] = False
ev = Evaluator(X=df, y=df[label_col_2_predict], model=classifier, feat_cols_and_hyperparams=ex_params)
ev.classic_eval(split_on_col='is_train_set')

# if model_type == 'SVM':
#     analysis_df = ev.get_feature_analysis_dataframe() #Columns: NEITHER, SINK, SOURCE
#     print(analysis_df['SOURCE'].sort_values(ascending=False))