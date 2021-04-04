import numpy as np
import pandas as pd
from utils.FormattingTools import Preprocessor
from utils.Configuration import ConfigSVM, ConfigNLFeaturesSearch, ConfigSVMSearch
from Evaluation.Evaluator import Evaluator
from SSModel.cLiSSAfier import cLiSSAfier
from SSModel.InternalSSVectorizers.BoWVectorizer import BoWVectorizer
import sys
from collections import defaultdict
import pickle
from pathlib import Path


'''
This experiments compares features on the OLD set of method annotations (api level 18 and below) only
10 fold NESTED cross-validation
'''

EXPERIMENT_NAME="NLFeatsOnly"
cached_file = None#'Inputs/Caches/finalized_exp1_cache.pickle'
annotation_set_fname = "Inputs/FinalizedAnns_4_2_APIs.csv"
docs_fname = "Inputs/android.csv"
man_feats_fname = "Inputs/Caches/DocOnlyManFeats.pickle"#"Inputs/manual_features.pickle""Inputs/SuSiAnnFeats"#
fname_to_store_cache = 'Inputs/Caches/finalized_exp1_cache.pickle'#'Inputs/Caches/susi_anns_cache.pickle'
experiment_record_savedir = Path("SuSiResults_exp1_nested/"+EXPERIMENT_NAME)
label_col_2_predict = 'Annotation'
model_type = 'SVM'
inner_folds = 10
outer_folds = 10
runs_to_average = 5
stochastic = True

## Features to optimize in inner loop. Comment out to remove feature set from model
feats2search_spaces = {}
# feats2search_spaces["Manual_Feats"] = [True, False]
feats2search_spaces["Return"] = ConfigNLFeaturesSearch(
                                          min_df_lst=[1, 3, 6, 9]
                                        , max_df_lst=[.3, .5,.7,.9]
                                        , max_features_lst=[1000, 2000, 4000, 8000]
                                        , ngram_range_lst=[(1,1), (1,2), (1,3)]
                                        )
feats2search_spaces["Parameters"] = ConfigNLFeaturesSearch(
                                          min_df_lst=[1, 3, 6, 9]
                                        , max_df_lst=[.3,.5,.7,.9]
                                        , max_features_lst=[1000, 2000, 4000, 8000]
                                        , ngram_range_lst=[(1,1), (1,2), (1,3)]
                                        )
feats2search_spaces["SigFeatures"] = ConfigNLFeaturesSearch(
                                          min_df_lst=[1, 3, 6, 9]
                                        , max_df_lst=[.3, .5, .7, .9]
                                        , max_features_lst=[1000, 2000, 4000, 8000]
                                        , ngram_range_lst=[(1,1), (1,2), (1,3)]
                                        )
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


#Preprocessing step (this is customized for our files)
if cached_file is None:
    preprocessor = Preprocessor(anns_fname=annotation_set_fname, docs_fname=docs_fname, man_feats_fname=man_feats_fname)
    df = preprocessor.preprocess_data()
    if fname_to_store_cache is not None:
        df.to_pickle(fname_to_store_cache)
else:
    df = pd.read_pickle(cached_file)

print_stats(df)
print("Running experiment:", EXPERIMENT_NAME)
print("Outer folds:", outer_folds)
print("Inner folds:", inner_folds)

### Check manual features active in dataset ###
# bow = BoWVectorizer()
# mf_col = bow.find_man_feat_cols(df)
# print(len(mf_col))
# print(df[mf_col].sum(axis=0).value_counts().sort_index())
# col_cnts = df[mf_col].sum(axis=0)
# print(col_cnts.loc[col_cnts <= 0.0].shape)

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

print("Final Result:")
print(perfomance_estimate)
print("Standard dev.:")
print(full_record['std'])

def save_experiment_results(results:pd.DataFrame, full_record:dict, save_dir:Path):
    results_path = save_dir / "results.pickle"
    full_record_path = save_dir / "full_record.pickle"
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    results.to_pickle(results_path)
    with open(full_record_path, 'wb') as f:
        pickle.dump(full_record, f)


is_valid_dirname = False
while experiment_record_savedir.exists() and not is_valid_dirname:
    if experiment_record_savedir.exists():
        answer = input("\nDirectory: "+str(experiment_record_savedir)+" exists. Do you want to overwrite? [y/n] ").lower().strip()

    if len(answer) > 1:
        answer = answer[0]
    elif len(answer) == 0:
        print("Must enter y or n. Try again")
        continue
    if answer == 'y':
        break
    elif answer == 'n':
        while True:
            experiment_record_savedir = Path(input("Enter new directory: ").strip())
            try:
                experiment_record_savedir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print("Error occured:")
                print(e)
                print("Try again")
                continue
            break
    else:
        print("Answer:", answer, "is invalid. Try again")
        continue


save_experiment_results(perfomance_estimate, full_record, experiment_record_savedir)
    

# if model_type == 'SVM':
#     analysis_df = ev.get_feature_analysis_dataframe() #Columns: NEITHER, SINK, SOURCE
#     pd.set_option('display.max_colwidth', 1000)
#     pd.set_option('display.min_rows', 50)
#     print(analysis_df[['SINK','SOURCE']].abs().sum(axis=1).sort_values(ascending=False))
#     print((analysis_df[['SINK','SOURCE']].abs().sum(axis=1) < 0.000001).sum())
#     null_svm_weights= analysis_df[['SINK','SOURCE']].index[(analysis_df.abs().sum(axis=1) < 0.000001)].tolist()
#     print(null_svm_weights)
#
#     feat_cols = df[mf_col].sum(axis=0)
#     print(feat_cols.index[feat_cols <= 1.0])
#     feat_cols[feat_cols <= 1.0].to_csv("in_feat_domain_no_feats.csv")












