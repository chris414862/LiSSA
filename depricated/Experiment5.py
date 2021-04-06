import numpy as np
import pandas as pd
from utils.FormattingTools import Preprocessor
from Evaluation.Evaluator import Evaluator
from SSModel.cLiSSAfier import cLiSSAfier
import sys

'''
This experiments uses 10 fold cross-validation to train and predict over ALL of our annotations together
'''

cached_file = None#'Inputs/Caches/finalized_exp5_cache.pickle'
annotation_set_fname = "Inputs/FinalizedAnns_4_2_APIs.csv"
annotation_set_fname2 = "Inputs/FinalizedAnns_NewAPIs.csv"
docs_fname = "Inputs/android.csv"
man_feats_fname = "Inputs/Caches/DocOnlyManFeats.pickle"#"Inputs/manual_features.pickle""Inputs/SuSiAnnFeats"#
fname_to_store_cache = 'Inputs/Caches/finalized_exp5_cache.pickle'#'Inputs/Caches/susi_anns_cache.pickle'
label_col_2_predict = 'Annotation'
remove_no_man_feats = True
model_type = 'SVM'
num_folds = 10
runs_to_average = 5

#Experimental parameters for SVM
ex_params = {
    # 'Manual_Feats': True,

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

def print_stats(df=None):
    print("From annotations:")
    print("Percent of sources:"
          , df.loc[ (df[label_col_2_predict] =='SOURCE')].shape[0]/df.loc[ df[label_col_2_predict].notna()].shape[0]
          , 'total:', (df[label_col_2_predict] == 'SOURCE').sum())

    print("Percent of sinks:"
          ,df.loc[ (df[label_col_2_predict] =='SINK')].shape[0]/df.loc[ df[label_col_2_predict].notna()].shape[0]
          , 'total:', (df[label_col_2_predict] == 'SINK').sum())
    print("Percent of neither:"
          ,df.loc[ (df[label_col_2_predict] =='NEITHER')].shape[0]/df.loc[ df[label_col_2_predict].notna()].shape[0]
          , 'total:', (df[label_col_2_predict] == 'NEITHER').sum())


#Preprocessing step (this is customized for our files)
if cached_file is None:
    preprocessor = Preprocessor(anns_fname=annotation_set_fname, docs_fname=docs_fname, man_feats_fname=man_feats_fname)
                                #, remove_no_man_feats=remove_no_man_feats)
    df = preprocessor.preprocess_data()
    preprocessor = Preprocessor(anns_fname=annotation_set_fname2, docs_fname=docs_fname, man_feats_fname=man_feats_fname)
                                #, remove_no_man_feats=remove_no_man_feats)
    df2 = preprocessor.preprocess_data()
    # preprocessor = Preprocessor(anns_fname=None, docs_fname=docs_fname,
    #                             man_feats_fname="Inputs/SuSiAnnFeats",
    #                             use_man_feat_anns=True)
    # df3 = preprocessor.preprocess_data()
    df = df.append(df2)
    # df = df.append(df3)
    if fname_to_store_cache is not None:
        df.to_pickle(fname_to_store_cache)
else:
    df = pd.read_pickle(cached_file)

print_stats(df)
classifier = cLiSSAfier(model_type)

ev = Evaluator(X=df, y=df[label_col_2_predict], model=classifier, feat_cols_and_hyperparams=ex_params
               , runs_to_average=runs_to_average)
ev.kfold_eval(folds=num_folds)

