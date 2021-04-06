import pandas as pd
from utils.FormattingTools import preprocess_data



testing_params = {}
#What feature types to use
#Documentation features
testing_params["use_doc_feats"] = True
testing_params["use_ret_feats"] = True
testing_params["use_param_feats"] = True
#Manual features
testing_params["use_man_feats"] = False
#Features from method signature
testing_params["use_sig_feats"] = True

#Features from method signature (breaks camel case and removes ".")
testing_params["add split class name"] = testing_params["use_sig_feats"]
testing_params["add split qual path"] = testing_params["use_sig_feats"]
testing_params["add split method name"] = testing_params["use_sig_feats"]

#Experiment type
testing_params["task"] = 'source/sink'
save_name = "doc_plus_sig_categories.csv"
cache_after_preprocessing = False

#input_file names
input_fnames = {'anns_fname'            : 'Inputs/new_anns2.csv',
                'docs_fname'            : '',
                'manual_feats_fname'    : ''
                }

cached_preprocessed_fname = ''

if cached_preprocessed_fname is not None and cached_preprocessed_fname != '':
    df = pd.read_pickle(cached_preprocessed_fname)
else:
    df = preprocess_data(input_fnames)
    if cache_after_preprocessing is not None:
        df.to_pickle(cached_preprocessed_fname)


pd.set_option('display.max_colwidth', -1)