from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from SSModel.ModelInterface import Model
import sys
from scipy.sparse.csr import csr_matrix
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
        # self.row_idx_labels: pd.Index= row_idx_labels
        # self.pred_labels = pred_labels
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
        return self.average_classification_reports(reports)

    def create_classification_report_df(self, cr, y_hat):

        cr.pop('accuracy')
        df = pd.DataFrame.from_dict(cr, orient='index')
        df['#Predictions'] = pd.NA
        vc = y_hat.value_counts()
        df.loc[df.index.isin(vc.index), '#Predictions'] = vc
        return df



    def average_classification_reports(self, reports: list, ytest_hat=None):
        '''
        Performs an averaging operation over the field in a list of multilevel dicts 'reports'. 'reports' should be
        a list of return values from sklearn's classification_report.

        :param reports:
        :param ytest_hat:
        :return:
        '''
        counts = {cl:0 for cl in self.class_labels}
        max_len_index = max([rep.index for rep in reports], key=lambda x: x.shape[0])
        sum_df = pd.DataFrame(index=max_len_index, columns=["precision", "recall", "f1-score", "support", "#Predictions"])
        sum_df = sum_df.fillna(0.0)
        # print('tracking reports')
        for i,rep in enumerate(reports):
            for row in rep.index:
                if row in self.class_labels and row in rep.index.tolist()\
                        and not (rep.loc[row,'support'] ==0.0 and  rep.loc[row,'#Predictions']==0.0):
                    counts[row] +=1
            # print(i)
            # print(rep.fillna(0.0))
            sum_df.loc[rep.index] += rep.fillna(0.0)
            # print(sum_df)

        for count_key in counts.keys():
            sum_df.loc[count_key] /= counts[count_key]
        sum_df.loc['macro avg'] /= len(reports)
        sum_df.loc['weighted avg'] /= len(reports)
        sum_df.loc[['macro avg','weighted avg'],'#Predictions'] = pd.NA
        # print('counts:', counts)
        print("####### Final Result ###########")
        print(sum_df)

        return sum_df



class Evaluator():

    def __init__(self, *args, **kwargs):
        self.evaluation = Evaluation(*args, **kwargs)

    def kfold_eval(self, *args, **kwargs):
        self.evaluation.kfold_eval(*args, **kwargs)

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


