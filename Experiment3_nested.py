import numpy as np
import pandas as pd
from utils.FormattingTools import Preprocessor
from utils.Configuration import ConfigSVM, ConfigNLFeaturesSearch, ConfigSVMSearch
from utils.SaveUtils import store_experiment_results
from utils.terminalsize import get_terminal_size
from evaluation.Evaluator import Evaluator
from SSModel.cLiSSAfier import cLiSSAfier
import sys
import os
import re
import subprocess
from pathlib import Path


'''
This experiments uses 10 fold cross-validation to train and predict over ALL of our annotations together
'''

#Experimental parameters
EXPERIMENT_NAME="NLFeatsOnly" #Used to control which feature set to use AND which directory to save to
feature_sets = {
                "AllFeats"        : ["Manual_Feats", "Return", "Parameters", "SigFeatures", "Description"],
                "NLFeatsOnly"     : ["Return", "Parameters", "SigFeatures", "Description"],
                "ManualFeatsOnly" : ["Manual_Feats"],
                }
feature_sets_to_include = feature_sets[EXPERIMENT_NAME] 
model_type = 'SVM'
inner_folds = 10
outer_folds = 10
runs_to_average = 5
stochastic = True #can use for debugging and/or replicating results
label_col_2_predict = 'Annotation' #included for added flexibility


#Data directories
cached_file = None#'Inputs/Caches/finalized_exp5_cache.pickle'
annotation_set_fname = "inputs/FinalizedAnns_4_2_APIs.csv"
annotation_set_fname2 = "inputs/FinalizedAnns_NewAPIs.csv"
docs_fname = "inputs/android.csv"
man_feats_fname = "inputs/Caches/DocOnlyManFeats.pickle"#"Inputs/manual_features.pickle""Inputs/SuSiAnnFeats"#
# man_feats_fname2 = "Inputs/manual_features.pickle"#"Inputs/SuSiAnnFeats"#
fname_to_store_cache = 'inputs/Caches/finalized_exp3_cache.pickle'#'Inputs/Caches/susi_anns_cache.pickle'

#Path for saving results
results_subdir = re.sub(r"\..*", "", __file__)
experiment_record_savedir = Path("./results/", results_subdir, EXPERIMENT_NAME)

## Features to optimize in inner loop. Comment out to remove feature set from model and optimization training
feats2search_spaces = {}
if "Manual_Feats" in feature_sets_to_include:
    feats2search_spaces["Manual_Feats"] = [True, False]
if "Return" in feature_sets_to_include:
    feats2search_spaces["Return"] = ConfigNLFeaturesSearch(
                                              min_df_lst=[1, 3, 6, 9]
                                            , max_df_lst=[.3, .5,.7,.9]
                                            , max_features_lst=[1000, 2000, 4000, 8000]
                                            , ngram_range_lst=[(1,1), (1,2), (1,3)]
                                            )
if "Parameters" in feature_sets_to_include:
    feats2search_spaces["Parameters"] = ConfigNLFeaturesSearch(
                                              min_df_lst=[1, 3, 6, 9]
                                            , max_df_lst=[.3,.5,.7,.9]
                                            , max_features_lst=[1000, 2000, 4000, 8000]
                                            , ngram_range_lst=[(1,1), (1,2), (1,3)]
                                            )
if "SigFeatures" in feature_sets_to_include:
    feats2search_spaces["SigFeatures"] = ConfigNLFeaturesSearch(
                                              min_df_lst=[1, 3, 6, 9]
                                            , max_df_lst=[.3, .5, .7, .9]
                                            , max_features_lst=[1000, 2000, 4000, 8000]
                                            , ngram_range_lst=[(1,1), (1,2), (1,3)]
                                            )
if "Description" in feature_sets_to_include:
    feats2search_spaces["Description"] = ConfigNLFeaturesSearch(
                                              min_df_lst=[1, 3, 6, 9]
                                            , max_df_lst=[.3, .5,.7,.9]
                                            , max_features_lst=[1000, 2000, 4000, 8000, 12000]
                                            , ngram_range_lst=[(1,1), (1,2), (1,3)]
                                            )


def print_stats(df=None):
    print("From annotations:")
    print("Percent of sources:"
          , df.loc[ (df[label_col_2_predict] =='SOURCE')].shape[0]/df.loc[ df[label_col_2_predict].notna()].shape[0])
    print("Percent of sinks:"
          ,df.loc[ (df[label_col_2_predict] =='SINK')].shape[0]/df.loc[ df[label_col_2_predict].notna()].shape[0])
    print("Percent of neither:"
          ,df.loc[ (df[label_col_2_predict] =='NEITHER')].shape[0]/df.loc[ df[label_col_2_predict].notna()].shape[0])
    print("Running experiment:", EXPERIMENT_NAME)
    print("Outer folds:", outer_folds)
    print("Inner folds:", inner_folds)
    print("Features considered:")
    [print("\t\t",feat_set_name) for feat_set_name in feature_sets_to_include] 

#Preprocessing step (this is customized for our files)
if cached_file is None:
    print("\nProcessing first annotation set...")
    preprocessor = Preprocessor(anns_fname=annotation_set_fname, docs_fname=docs_fname, man_feats_fname=man_feats_fname)
    df = preprocessor.preprocess_data()
    
    # Reprocessing docs and manual features for implementation ease. Most of runtime spent in training
    print("\nProcessing second annotation set...")
    preprocessor = Preprocessor(anns_fname=annotation_set_fname2, docs_fname=docs_fname, man_feats_fname=man_feats_fname)
    df2 = preprocessor.preprocess_data()

    df = df.append(df2)
    if fname_to_store_cache is not None:
        df.to_pickle(fname_to_store_cache)
else:
    df = pd.read_pickle(cached_file)

print_stats(df)
classifier = cLiSSAfier(model_type)

evaluator = Evaluator(X=df, y=df[label_col_2_predict], model=classifier#, feat_cols_and_hyperparams=ex_params
               , runs_to_average=runs_to_average)

perfomance_estimate, full_record =  evaluator.average_multiple(  
                                                                  num_runs=runs_to_average
                                                                , func=evaluator.nested_kfold_eval 
                                                                , func_name="nested cross-validation"
                                                                , stochastic=stochastic
                                                                , func_params={
                                                                                 "inner_folds" : inner_folds
                                                                               , "outer_folds" : outer_folds
                                                                               , "feats2search_spaces" : feats2search_spaces
                                                                              }
                                                                )
full_record['feats_considered']=feature_sets_to_include

print("Final Result:")
print(perfomance_estimate)
print("Standard dev.:")
print(full_record['std'])



store_experiment_results(experiment_record_savedir, perfomance_estimate, full_record)
