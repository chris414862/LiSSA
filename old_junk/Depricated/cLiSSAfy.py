import argparse
import pandas as pd
from utils.FormattingTools import preprocess_data, add_toks_in_signature
from OldJunk.Depricated.UpdateSimulator import run_test
import random
from OldJunk.Depricated.UpdateSimulator import average_classification_reports


def parse_args()-> argparse.Namespace:
    parser = argparse.ArgumentParser("This tool can be used to train new ML classifier models for categorizing\n"
                                     +" (Java, Android) methods as either sources and sinks, then further classifies\n"
                                      +" them into them"
                             " into classes ")
    parser.add_argument("-manFeats", help="path to directory or file that holds the manual feature files")
    parser.add_argument("-annFile", help="path to file with annotations")
    parser.add_argument("-docs", help="path do file with method documentation")
    parser.add_argument("-model", help='Include a model if you want to do inference on a trained model')
    parser.add_argument("-cache_preprocessing"
                        ,help='Store preprocessed and aggregated data in a pandas DataFrame for easy rerunning.'+
                        " Indicate file name after -cache_preprocessing flag.")
    parser.add_argument("-cached_file"
                        , help='Use file that has already been preprocessed.')

    return parser.parse_args()



def main():
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
    testing_params["simulate_updates"] = True
    testing_params["return_predict"] = False
    testing_params["evaluate"] = not testing_params["return_predict"]
    testing_params["diff_testing_population"] = False
    testing_params["dsafe_only"] = True
    testing_params["save_random_sample"] = False
    testing_params["n_samples"] = 15
    testing_params["add_new_anns"] = True
    testing_params["pct_new_test"] = .75
    testing_params["kfold"] = False
    testing_params["num_folds"] = 10
    testing_params['only_new'] = False
    testing_params['repeats'] = 5
    # "all_doc_no_man.csv", "only_man_feats.csv", "doc_plus_man.csv","doc_plus_sig.csv"
    save_name = "doc_plus_sig_categories.csv"
    new_anns_fname = 'Inputs/new_anns2.csv'

    args = parse_args()
    if args.cached_file is not None:
        df = pd.read_pickle(args.cached_file)
    else:
        df = preprocess_data((args.annFile, args.docs, args.manFeats))
        if args.cache_preprocessing is not None:
            df.to_pickle(args.cache_preprocessing)
    pd.set_option('display.max_colwidth', -1)

    #Add in annotations from API 28,29
    if testing_params["add_new_anns"]:
        print("Adding new annotations")
        new_anns = pd.read_csv(new_anns_fname, index_col=0)
        new_anns["ApiLevel"] = pd.to_numeric(new_anns["ApiLevel"], errors='coerce')
        df.loc[new_anns.index,"ApiLevel"] = new_anns["ApiLevel"]
        df.loc[new_anns.index,"Origin"] = 'new'
        df.loc[new_anns.index,"Source/Sink"] = new_anns["SS GroundT"]
        # df.loc[(df["ApiLevel"]>27 )& (df["Origin"] != 'new'), 'Origin'] = pd.NA
        df = df.loc[~((df["ApiLevel"] > 27) & (df["Origin"] != 'new'))]
        print("CHECK")
        print(df.loc[df["ApiLevel"] >27].shape)
        print((df['Origin']=='new').sum())








    df = df.loc[~df.index.duplicated(keep='first')]
    df["Source/Sink"][df["Source/Sink"].isna()] = 'none'
    if testing_params["dsafe_only"]:
        if testing_params["add_new_anns"]:
            df = df.loc[(df['Origin'] == 'dsafe')|(df['Origin']=='new')]
        else:
            df = df.loc[df['Origin'] == 'dsafe']


    print("Percent of sources (from annotations)")
    print(df.loc[ (df['Source/Sink'] =='source')].index.size/df.loc[ df["Source/Sink"].notna()].index.size)
    add_toks_in_signature(df, method_name=testing_params["add split method name"]
                          , class_name=testing_params["add split class name"]
                          , qual_path=testing_params["add split qual path"])

    if testing_params["diff_testing_population"] == True:
        testing_params["diff_testing_population"] = df.loc[
                                    df["ApiLevel"].notna() & df["Origin"].isna()]
    else:
        testing_params["diff_testing_population"] =pd.DataFrame({"a":[]})


    ann_w_doc = df.loc[df["ApiLevel"].notna() & df["Origin"].notna()]
    ann_w_doc = ann_w_doc.loc[ann_w_doc["Source/Sink"] != 'unannotated']
    ann_w_doc = ann_w_doc.loc[~ann_w_doc.index.duplicated(keep='first')]

    # Levels made explicit in dict for readability
    update_sets = [
                    {  #Iteration 1
                        "train levels":(1,18)
                        , "test levels":(28,29)
                    },
                    #{ #Iteration 2
                    #   "train levels": (1, 8)
                    #   , "test levels": (9, 12)
                    #}
                     ]
    ann_w_doc_orig = ann_w_doc.copy()
    # if testing_params["simulate_updates"]:
    rets = []
    runs2avg = testing_params['repeats']
    for train_test_spec in update_sets:
        for run in range(runs2avg):
            print("Run:", run)
            if testing_params["pct_new_test"] > 0.0:
                print("Before split:")
                print(ann_w_doc_orig.loc[ann_w_doc_orig["ApiLevel"] > 27].shape)
                ann_w_doc = ann_w_doc_orig.copy()
                new_anns = pd.read_csv(new_anns_fname, index_col=0)
                test = new_anns.sample(frac=testing_params["pct_new_test"])
                train = new_anns.drop(test.index, inplace=False)

                ann_w_doc.loc[train.index, "ApiLevel"] = 1
                print("After split:")
                print(ann_w_doc.loc[ann_w_doc["ApiLevel"] > 27].shape)
                if testing_params['only_new']:
                    ann_w_doc = ann_w_doc.loc[ann_w_doc["ApiLevel"] > 27]
            random.seed()
            seed = random.randint(0,1000)
            print(seed)
            ret,ytest_hat = run_test(ann_w_doc.sample(frac=1.0, random_state=seed), train_test_spec, testing_params)
            rets.append(ret)
    if not testing_params["kfold"]:
        average_classification_reports(rets, ytest_hat=ytest_hat)



        # print(rets[0] ["Yhat"].value_counts())
        # print(rets[0]["Yhat"].value_counts()/rets[0]["Yhat"].index.size)
        # print(rets[0].loc[rets[0] ["Yhat"]=="NETWORK"])
        # print(rets[0].loc[rets[0]["Yhat"] == "AUDIO"])
        # results = mts

        # df = df.loc[~df.index.duplicated(keep='first')]
        #
        # format_and_save_csv(results, df, save_name)
        # if testing_params["save_random_sample"]:
        #     mask = results.loc[results["Yhat"]=="source"].sample(n=testing_params["n_samples"]).index.to_series()
        #     mask = mask.append(results.loc[results["Yhat"]=="sink"].sample(n=testing_params["n_samples"]).index.to_series())
        #     mask = mask.append(results.loc[results["Yhat"] == "none"].sample(n=testing_params["n_samples"]).index.to_series())
        #     results_sample = results.loc[mask].sample(frac=1.0, axis=0)
        #     results_sample_blind = results_sample.drop("Yhat", axis=1)
        #     save_name_sample = re.sub(r"(.*)\.csv$", r"\1_SAMPLED_BLIND.csv", save_name)
        #     format_and_save_csv(results_sample_blind, df, save_name_sample, sort=False)
        #
        #     save_name_sample = re.sub(r"(.*)\.csv$", r"\1_SAMPLED_W_YHAT.csv", save_name)
        #     format_and_save_csv(results_sample, df, save_name_sample)

    #Classic train test split of all annotations (70-30 split)
    # else:
    #     if len(update_sets) ==0:
    #         print("Bug: Fix update_sets")
    #         sys.exit()
    #     ret = run_test(ann_w_doc, update_sets[0], testing_params)




if __name__ == "__main__":
    main()



