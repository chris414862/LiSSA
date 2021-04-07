from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from SSModel.ModelInterface import Model
import sys
import os
from scipy.sparse.csr import csr_matrix
from utils.Configuration import ConfigSVMSearch, ConfigNLFeatures, ConfigNLFeaturesSearch, ConfigSVM
import random
import time
'''
This class is responsible for performing the evaluation of a cLiSSAfier model. The internals of the classification 
model, tokenization scheme, and the vector representations are abstracted and lie in the implementation of the 
cLiSSAfier class. This class performs classic train/test split evaluation as well as kfold evaluation.
'''



class Evaluation():
    def __init__(self, X:pd.DataFrame, y:pd.Series, model:Model, row_idx_labels=None
                 # , pred_labels=None
                 , feat_cols_and_hyperparams=None, X_tilde:pd.DataFrame=None
                 , feat_cols_and_hyperparams_tilde=None
                 , add_to_train_fold_X = None
                 , add_to_train_fold_y = None
                 , runs_to_average = 1
                 , use_man_feat_anns=False):
        self.X: pd.DataFrame = X
        self.y: pd.DataFrame = y
        self.model: Model = model.get_internal_model()
        self.feat_cols_and_hyperparams=feat_cols_and_hyperparams
        self.feat_cols_and_hyperparams_tilde = feat_cols_and_hyperparams_tilde
        self.X_tilde = X_tilde
        self.use_self_learning = X_tilde is not None
        self.add_to_train_fold_X = add_to_train_fold_X
        self.add_to_train_fold_y = add_to_train_fold_y
        self.runs_to_average = runs_to_average
        self.class_labels = self.y.unique().tolist()
        self.use_man_feat_anns = use_man_feat_anns



    def classic_eval(self, rand_split_pct=None, split_on_col=None):
        # self.model.train(self.X, self.y, feat_cols_and_hyperparams==self.feat_cols_and_hyperparams)
        if rand_split_pct is not None and split_on_col is not None:
            print("Logic error. rand_split_pct and split_on_col cannot both be set.")

        if rand_split_pct is not None:
            X_train, X_test = (
            self.X.sample(frac=rand_split_pct, axis=0), self.X.sample(frac=1 - rand_split_pct, axis=0))

        elif split_on_col is not None:
            X_train = self.X.loc[self.X[split_on_col]].sample(frac=1.0)
            X_test = self.X.loc[~self.X[split_on_col]].sample(frac=1.0)


        y_train, y_test = (self.y[X_train.index], self.y[X_test.index])

        self.model.train(X_train, y_train, feat_cols_and_hyperparams=self.feat_cols_and_hyperparams)
        if self.use_self_learning:
            y_test_hat = self.model.predict(X_test)
            y_train_hat = self.model.predict(X_train)
            print('(before self-learning) Training set:')
            print('Number of predictions:')
            print(y_train_hat.value_counts())
            print(classification_report(y_train, y_train_hat, digits=4, zero_division=0))
            print('(before self-learning)Testing set:')
            print('Number of predictions:')
            print(y_test_hat.value_counts())
            print(classification_report(y_test, y_test_hat, digits=4, zero_division=0))
            y_train_tilde,_ = self.model.predict(self.X_tilde)
            y_train = y_train.append(y_train_tilde)
            X_train = X_train.append(self.X_tilde)
            print(self.feat_cols_and_hyperparams_tilde)
            self.model.train(X_train, y_train, feat_cols_and_hyperparams=self.feat_cols_and_hyperparams_tilde)


        y_test_hat,_ = self.model.predict(X_test)
        y_train_hat,_ = self.model.predict(X_train)
        if self.use_self_learning:
            print('(after self-learning)Training set:')
        else:
            print('Training set:')

        cr = classification_report(y_train, y_train_hat, digits=4, output_dict=True, zero_division=0)
        cr_df = self.create_classification_report_df(cr, y_train_hat)
        print(cr_df)

        if self.use_self_learning:
            print('(after self-learning)Testing set:')
        else:
            print('Testing set:')
        cr = classification_report(y_test, y_test_hat, digits=4, output_dict=True, zero_division=0)
        cr_df = self.create_classification_report_df(cr, y_test_hat)
        print(cr_df)

    def perform_kf_split(X:pd.DataFrame, y:pd.Series, train_test_idxs:tuple):
        kf_train_idx, kf_test_idx = train_test_idxs
        kf_train_idx_labels = X.index[kf_train_idx]
        kf_test_idx_labels = X.index[kf_test_idx]
        x_train, y_train = X.loc[kf_train_idx_labels], y.loc[kf_train_idx_labels]
        x_test, y_test = X.loc[kf_test_idx_labels], y.loc[kf_test_idx_labels]
        return x_train, y_train, x_test, y_test

    def kfold_eval(self, folds=10):
        run_sum = None
        for i in range(self.runs_to_average):
            res = self.kfold_eval_internal(folds=folds)
            if run_sum is None:
                run_sum = res
            else:
                run_sum += res
        if self.runs_to_average > 1:
            print("!!!!!! After averaging",self.runs_to_average,"runs, final tally:")
            avg_df:pd.DataFrame = run_sum/self.runs_to_average

            avg_df = avg_df.reindex(['SOURCE','SINK','NEITHER','macro avg','weighted avg'])
            avg_df.index = ['Source','Sink', 'Neither', 'Macro Avg.', 'Weighted Avg.']

            print(avg_df)
            avg_df = avg_df.drop('support', axis=1)
            avg_df = avg_df.drop('#Predictions', axis=1)
            avg_df.columns = ['Precision', 'Recall', 'F1']
            print(avg_df.round(3).to_latex())


    def max_average(self,classification_df_lists, configs, silent=True):
        best_config = None
        best_score = -1.0
        stats_of_best = None
        if len(classification_df_lists) != len(configs):
            raise ValueError("classification_df_lists length should equal configs length. Instead:"+str(len(classification_df_lists))+", "+str(len(configs)))
        for i in range(len(configs)):
            config_results_list = classification_df_lists[i]
            config = configs[i]
            avg_result, std_df = self.average_classification_reports(config_results_list)

            if avg_result.loc['macro avg', 'f1-score'] >= best_score:
                best_score = avg_result.loc['macro avg', 'f1-score']
                best_config = config
                stats_of_best = (avg_result, std_df)
            if not silent:
                print(config)
                print(avg_result)
                print()

        return best_config, stats_of_best

    class inner_prog_bar():
        def __init__(self, tot_folds, sub_steps, bar_width=200):
            self.tot_folds = tot_folds
            self.num_sub_steps = sub_steps
            self.tot_steps = self.tot_folds*self.num_sub_steps
            self.label_string ="Inner fold number: {curr_fold_num} "
            self.max_label_string_len = len(self.label_string.format(curr_fold_num=self.tot_folds))
            self.curr_step = 0

            #-5 explanation: -2 from progbar border, -1 from arrow, -2 for terminal boarder padding
            self.bar_width = bar_width if bar_width < os.get_terminal_size().columns - self.max_label_string_len -5\
                    else os.get_terminal_size().columns - self.max_label_string_len -5

        def display(self):
            fold_num = self.curr_step//self.num_sub_steps+1
            progress = int(self.curr_step/self.tot_steps *self.bar_width)

            # Make sure last step completes bar. If bar_width > tot_steps, gap can be left
            if self.curr_step == self.tot_steps-1:
                #Leave one space for '>'
                progress = self.bar_width-1

            string = f"{self.label_string.format(curr_fold_num=fold_num):<{self.max_label_string_len}}"
            string += f" |{('='*progress)+'>':<{self.bar_width}}|"
            if self.curr_step < self.tot_steps - 1:
                string += "\r"
            else:
                string += "\n"
            print(string, end="")
            self.curr_step += 1


    def find_best_config(self, x_train, y_train, inner_cv, feature_column=None, search_space=None, seed=0):
        if search_space is None:
            return None
        best_config=None
        best_score=0.0
        configs =[] 
        classification_df_lists = []#[[] for i in range(num_configs)]
        prog_bar = self.inner_prog_bar(inner_cv.n_splits, search_space.num_to_generate())
        for j, inner_train_test_idxs in enumerate(inner_cv.split(x_train)):
            inner_x_train, inner_y_train, inner_x_dev, inner_y_dev = Evaluation.perform_kf_split(x_train, y_train, train_test_idxs=inner_train_test_idxs)

            for i, config in enumerate(search_space.generator()):
                prog_bar.display()
                if feature_column is None:# Used in feature subset optimization
                    feat_cols_and_hyperparams = self.svm_config2model_params(config)
                
                else: # Used in Vectorizer optimization of one feature set 
                    feat_cols_and_hyperparams = {feature_column:config.as_feat_dict()}


                if j == 0:
                    classification_df_lists.append([])
                    configs.append(config)

                self.model.reset_w_seed(seed)
                self.model.train(inner_x_train, inner_y_train, feat_cols_and_hyperparams=feat_cols_and_hyperparams)
                y_dev_hat,_ = self.model.predict(inner_x_dev)
                cr = classification_report(inner_y_dev, y_dev_hat, digits=4, output_dict=True, zero_division=0)
                cr_df = self.create_classification_report_df(cr, y_dev_hat)
                classification_df_lists[i].append(cr_df)

         
        best_config, stats = self.max_average(classification_df_lists, configs)#, silent=feature_column is not None)
        return best_config, stats

    def nl_config2nl_search(self, config:ConfigNLFeatures):
        d = config.as_feat_dict()
        return ConfigNLFeaturesSearch(**{k+"_lst":[v] for k, v in d.items()})


    def svm_config2model_params(self, config:ConfigSVM):
        tmp = {k:v for k, v in config.as_feat_dict().items() if v is not None}
        feat_cols_and_hyperparams = {k:v.as_feat_dict()   for k, v in tmp.items() if k != "Manual_Feats"} 
        if "Manual_Feats" in tmp.keys():
            feat_cols_and_hyperparams["Manual_Feats"] = tmp["Manual_Feats"]
        return feat_cols_and_hyperparams


    def average_multiple(self, num_runs, func, func_name="", stochastic=True, func_params={}):
        header = f" AVERAGING {num_runs} FULL {func_name.upper()} EVALUATIONS "
        eval_title = ' STARTING EVAL #{eval_num} seed: {seed} '
        border_len = 20

        if stochastic:
            random.seed(time.time())
            seeds = [random.randint(0,1000) for i in range(num_runs)]
        else:
            seeds = [i for i in range(num_runs)]

        #TODO: Fully handle case for boarders when eval_title is longer than header
        title_len = max(len(header), len(eval_title.format(eval_num = num_runs, seed = max(seeds))))+2
        full_header = f"{header:#^{2*border_len+title_len}}"
        full_header_len = len(full_header)
        print(full_header)
        print("Stochastic:", stochastic)
        print("Seeds:", seeds)

        results_df_list = []
        full_record = {}
        for i in range(num_runs):
            eval_header = eval_title.format(eval_num=i+1,seed=seeds[i])
            print(f"{eval_header:=^{full_header_len}}")

            func_params['seed'] = seeds[i]
            results, run_record = func(**func_params) 
            results_df_list.append(results)
            full_record["run_"+str(i+1)] = run_record

            print("\n"+"-"*full_header_len)
            print("EVAL",i+1, "RESULTS:")
            print(results)
            print("-"*full_header_len)

        results_of_evals_list = [data_dict["stats"] for eval_num, data_dict in full_record.items()]        
        averaged_performance_estimate_df, std_df = self.average_classification_reports(results_of_evals_list)
        full_record["stats"] = averaged_performance_estimate_df
        full_record["std"] = std_df
        full_record["num_runs"] = num_runs
        full_record["info_str"] = "This level of the record contains results from averaging "+str(num_runs)\
                                    +func_name\
                                    +(" " if func_name != "" else "")\
                                    +"evaluations"

        print(f"{'FINISHED':#^{full_header_len}}")
        return averaged_performance_estimate_df, full_record 

    def nested_kfold_eval(self, outer_folds=10, inner_folds=10, feats2search_spaces=None, man_feats_name="Manual_Feats", seed=0):
        """
              Optimize hyperparameter settings on inner cross-val loop (used as dev set)
                  -First, find best hyperparameters separately for each feature class's TfidfVectorizer parameters
                  -Second, find best subset of the total set of feature classes
             
              Current set of feature classes: { manual features from SuSi, return descriptions, parameter descriptions
                                              , method descriptions, signature features}
        """
        to_shuffle=True

        inner_cv = KFold(n_splits=inner_folds, shuffle=to_shuffle, random_state=seed)
        outer_cv = KFold(n_splits=outer_folds, shuffle=to_shuffle, random_state=seed)

        outer_cv_record_dict = {}
        for i, outer_train_test_idxs in enumerate(outer_cv.split(self.X)):
            print("\nStarting outer loop:", i+1)

            out_x_train, out_y_train, out_x_test, out_y_test = Evaluation.perform_kf_split(self.X, self.y, train_test_idxs=outer_train_test_idxs)
            if len(feats2search_spaces.keys()) > 1 or "Manual_Feats" not in feats2search_spaces.keys():
                #Optimize with xx_train, evaluate with xx_test

                inner_cv_record_dict = {}
                if feats2search_spaces is None:
                    return None

                # Optimize Sklearn's Vectorizer hyperparameters
                inner_cv_record_dict["nl_features_set"] = {}
                for feat_name, search_space in feats2search_spaces.items():
                    if feat_name != man_feats_name:# Not optimizing manual features. Assumed to already have been done by SuSi authors. We are just using as-is
                        print("Finding best setting for",feat_name,"features...")
                        best_config, (stats, std_df) = self.find_best_config(out_x_train, out_y_train, inner_cv, feature_column=feat_name
                                                                            , search_space=search_space, seed=seed)
                        inner_cv_record_dict["nl_features_set"][feat_name] ={"config":best_config, "stats":stats, "std":std_df}


                # prepare svm feature set search space
                feat_sets_to_search = {}
                if len(inner_cv_record_dict['nl_features_set'].keys()) > 0: #documentation featurs
                    for feat_name, data_dict in inner_cv_record_dict['nl_features_set'].items():
                        feat_sets_to_search[feat_name] = self.nl_config2nl_search(data_dict["config"])

                if man_feats_name in feats2search_spaces.keys(): #manual (SuSi) features
                    feat_sets_to_search[man_feats_name] = feats2search_spaces[man_feats_name]

                svm_config_search = ConfigSVMSearch(feat_dict=feat_sets_to_search, generate_singletons=True)

                # Optimize subset to use
                print("Finding best feature subset...")
                best_svm_config, (best_subset_stats, std_df) = self.find_best_config(out_x_train, out_y_train, inner_cv
                                                                                    , search_space=svm_config_search, seed=seed)
                inner_cv_record_dict["config"] = best_svm_config
                inner_cv_record_dict["stats"] = best_subset_stats
                inner_cv_record_dict["std"] = std_df 
                inner_cv_record_dict["num_inner_folds"] = inner_cv.n_splits
                inner_cv_record_dict["info_str"] = "This level of the record contains results from the hyperparameter "\
                                                + "optimization in the inner loop of "\
                                                + "nested cross-validation (used as dev/train splits)."\
                                                + "The results of the various averaged inner cross-validation optimization runs can "\
                                                + "be found here, but we do not store the results of each fold of the inner loop due to "\
                                                + "the large search space and number of optimization steps."
                                            
            else:
                feat_sets_to_search = {}
                feat_sets_to_search[man_feats_name] = feats2search_spaces[man_feats_name]
                svm_config_search = ConfigSVMSearch(feat_dict=feat_sets_to_search, generate_singletons=True)
                if svm_config_search.num_to_generate() == 1:
                    best_svm_config = [c for c in svm_config_search.generator()][0]
                    best_subset_stats = None
                    inner_cv_record_dict = None

                elif svm_config_search.num_to_generate() > 1:
                    raise ValueError("Something went wrong. Check code")
                elif svm_config_search.num_to_generate() == 0:
                    print("Could not produce SVM configuration. Check parameters") 
                    return None, None

 



            # Evaluate best performing inner cv configuration on outer test fold
            self.model.reset_w_seed(seed)
            self.model.train(out_x_train, out_y_train, feat_cols_and_hyperparams=self.svm_config2model_params(best_svm_config))
            y_test_hat,_ = self.model.predict(out_x_test, out_y_test)
            cr = classification_report(out_y_test, y_test_hat, digits=4, output_dict=True, zero_division=0)
            stats = self.create_classification_report_df(cr, y_test_hat)
            print("\nBest configuration found in inner cv:")
            print(best_svm_config)
            print("Averaged inner cv results:")
            print(best_subset_stats)
            print("Results on outer fold "+str(i+1)+":")
            print(stats)

            # Store results
            outer_cv_record_dict["outer_fold_"+str(i+1)] = {}
            outer_cv_record_dict["outer_fold_"+str(i+1)]["stats"] = stats
            outer_cv_record_dict["outer_fold_"+str(i+1)]["inner_cv_record"] = inner_cv_record_dict
            outer_cv_record_dict["outer_fold_"+str(i+1)]["info_str"] = "This level of the record contains results from testing "\
                                                                      +"the model on outer fold number "+str(i+1)+". Hyperparameters "\
                                                                      +"were optimized using all folds not including "+str(i+1)+" by "\
                                                                      +"using an inner loop of cross-validation. Note: No standard "\
                                                                      +"deviation information exists for this level since "+str(i+1)+" "\
                                                                      +"is the only test fold on this level of the record." 


        outer_stats_df_list = [data_dict["stats"] for fold_num, data_dict in outer_cv_record_dict.items()]        
        final_performance_estimate_df, std_df = self.average_classification_reports(outer_stats_df_list)
        outer_cv_record_dict["stats"] = final_performance_estimate_df
        outer_cv_record_dict["std"] = std_df
        outer_cv_record_dict["num_outer_folds"] = outer_cv.n_splits
        outer_cv_record_dict["info_str"] = "This level of the record contains results from the testing folds of the outer loop of "\
                                           +"nested cross-validation"

        return final_performance_estimate_df, outer_cv_record_dict
        
        


    def kfold_eval_internal(self, folds=10):
        '''
        Performs a kfold evaluation

        :param folds:
        :return:
        '''
        if folds < 1:
            print("Logic error. Should have more than one fold for k-fold validation")

        kf = KFold(n_splits=folds, shuffle=True)#, random_state=0)
        reports = []
        for i, train_test_idxs in enumerate(kf.split(self.X)):
            print("####STARTING FOLD:", i + 1)
            x_train, y_train, x_test, y_test = Evaluation.perform_kf_split(self.X, self.y, train_test_idxs=train_test_idxs)
            if self.add_to_train_fold_X is not None:
                x_train = x_train.append(self.add_to_train_fold_X)
                y_train = y_train.append(self.add_to_train_fold_y)
            self.model.train(x_train, y_train, feat_cols_and_hyperparams=self.feat_cols_and_hyperparams)
            y_test_hat,_ = self.model.predict(x_test, y_test)
            # y_train_hat = self.model.predict(x_train)
            # print('Training set:')
            # cr = classification_report(y_train, y_train_hat, digits=4, output_dict=True, zero_division=0)
            # cr_df = self.create_classification_report_df(cr, y_train_hat)
            # print(cr_df)



            cr = classification_report(y_test, y_test_hat, digits=4, output_dict=True, zero_division=0)
            cr_df = self.create_classification_report_df(cr, y_test_hat)
            # print('\nTesting set:')
            # print(cr_df)
            reports.append(cr_df)
        avg_df, std_df = self.average_classification_reports(reports)
        return avg_df

    def create_classification_report_df(self, cr, y_hat):

        cr.pop('accuracy')
        df = pd.DataFrame.from_dict(cr, orient='index')
        df['#Predictions'] = pd.NA
        vc = y_hat.value_counts()
        df.loc[df.index.isin(vc.index), '#Predictions'] = vc
        return df


    def reduce_cr_list(self, reports, reduction="avg", avg_df=None):
        """
            Manually wrote reduction func because we want to make sure we handle edge cases correctly.
            Sometimes a fold has no instances of one class and we should not include that in the reduction
            if there were no predictions for it.
        """
        if len(reports) == 0:
            return None
        
        counts = {cl:0 for cl in self.class_labels}
        max_len_index = max([rep.index for rep in reports], key=lambda x: x.shape[0])
        reduc_df:pd.DataFrame = pd.DataFrame(index=max_len_index, columns=reports[0].columns)
        reduc_df = reduc_df.fillna(0.0)

        for i,report in enumerate(reports):

            #Only count cases when test fold had positive instances or there were no positive instances and our model predicted incorrectly
            # Some unlucky draws do not contain a class and if we do not track, we can over divide
            for row in report.index:
                if row in self.class_labels and not (report.loc[row,'support'] ==0.0 and  report.loc[row,'#Predictions']==0.0):
                    counts[row] +=1

            if reduction == "avg":
                reduc_df.loc[report.index] += report.fillna(0.0)
            elif reduction == "std" or reduction == "var":
                if avg_df is None:
                    raise ValueError("If reduction is std or var, must include avg_df")
                # print("report:")
                # print(report.fillna(0.0))
                # print("avg:")
                # print(avg_df)
                reduc_df.loc[report.index] += (report.fillna(0.0)-avg_df)**2

        # divide each row by correct number of instances
        for count_key in counts.keys():
            reduc_df.loc[count_key] /= counts[count_key]

        reduc_df.loc['macro avg'] /= len(reports)
        reduc_df.loc['weighted avg'] /= len(reports)

        if reduction == "std":
            reduc_df = reduc_df.apply(np.sqrt)

        return reduc_df


    def average_classification_reports(self, reports: list, ytest_hat=None):
        '''
        Performs an averaging operation over the field in a list of multilevel dicts 'reports'. 'reports' should be
        a list of return values from sklearn's classification_report.

        :param reports:
        :param ytest_hat:
        :return:
        '''
        metric_names = ["precision", "recall", "f1-score", "support", "#Predictions"]
        avg_df = self.reduce_cr_list(reports, reduction="avg")
        std_df = self.reduce_cr_list(reports, reduction="std", avg_df=avg_df)
        std_df.columns = [col_name+"(std)" for col_name in std_df.columns]
        return avg_df, std_df




class Evaluator():
    '''
        This basically just wraps the Evaluation class. However, the Evaluator is responsible 
        for feature analysis that is done after the evaluation.

    '''

    def __init__(self, *args, **kwargs):
        self.evaluation = Evaluation(*args, **kwargs)

    def kfold_eval(self, *args, **kwargs):
        self.evaluation.kfold_eval(*args, **kwargs)

    def nested_kfold_eval(self, *args, **kwargs):
        return self.evaluation.nested_kfold_eval(*args, **kwargs)

    def average_multiple(self, *args, **kwargs):
        return self.evaluation.average_multiple(*args, **kwargs)

    def classic_eval(self, *args, **kwargs):
        self.evaluation.classic_eval(*args, **kwargs)

    def get_index_labels(self):
        return self.evaluation.model.get_vectorizer().idx2feat

    def get_weights(self):
        '''
        Returns weights of model. Shape depends on internal model representation.

        Internal model      Weight shape
        SVM:                (1, n_features) if n_classes == 2, else (n_classes, n_features)
        :return:
        '''
        return self.evaluation.model.get_weights()

    def get_feature_analysis_dataframe(self):
        if self.evaluation.model.get_model_type() != 'SVM':
            print("Feature analysis is only available for the SVM model")
            return None

        else:
            idx2feat = self.get_index_labels()
            index = [idx2feat[idx] for idx in sorted(idx2feat.keys())]
            w = self.evaluation.model.get_weights()

            print('weight mat:', w.shape, w.T.shape)
            print(self.evaluation.model.get_class_labels())
            if type(w) != np.ndarray:
                w = w.todense()

            #np.absolute(w.T).sum(axis=1)
            return pd.DataFrame(w.T, index=index,columns=self.evaluation.class_labels)#.sort_values('Weights',ascending=False)


