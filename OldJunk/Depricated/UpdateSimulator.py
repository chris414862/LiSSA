import pandas as pd
from utils.SKLearnPrep import vectorize_methods
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import sys
from sklearn.model_selection import KFold
import random


def run_test(data:pd.DataFrame, ranges:dict, testing_params:tuple):
    print("LABEL COUNTS AND DISTRIBUTION:")
    if testing_params["task"] == 'category':
        #Remove catagories with too few samples
        too_few_list = data['Category'].value_counts().index[data['Category'].value_counts() < 20].to_series()
        data['Category'] = data['Category'].where(~data['Category'].isin(too_few_list), "NO_CATEGORY")
    else:
        print(data['Source/Sink'].value_counts())
        print(data['Source/Sink'].value_counts()/data.shape[0])
    print("#########################################################")

    (train_index, test_index), (Xtrain, Ytrain, Xtest,Ytest) = vectorize_methods(data
                                                                                 , train_levels=ranges["train levels"]
                                                                                 , test_levels=ranges["test levels"]
                                                                                 , use_doc_feats=testing_params["use_doc_feats"]
                                                                                 , use_ret_feats=testing_params["use_ret_feats"]
                                                                                 , use_param_feats=testing_params["use_param_feats"]
                                                                                 , use_man_feats=testing_params['use_man_feats']
                                                                                 , task=testing_params["task"]
                                                                                 , testing_pool= testing_params["diff_testing_population"]
                                                                                 , simulate_update=testing_params["simulate_updates"]
                                                                                 , use_sig_feats=testing_params["use_sig_feats"]
                                                                                 , kfold=testing_params['kfold'])

    print("Vector sizes:")
    print("train_index:", train_index.size,  "Xtrain:", Xtrain.shape,"Ytrain:", Ytrain.shape)
    if not testing_params['kfold']:
        print("test_index:",test_index.size, "Xtest:", Xtest.shape,"Ytest:", Ytest.shape)
    print("Starting training...")
    if testing_params["kfold"]:
        kfold_train_test(Xtrain,Ytrain,train_index, testing_params['num_folds'])
        return None, None
    else:
        random.seed()
        seed = random.randint(0, 1000)
        model = LinearSVC(random_state=seed).fit(Xtrain, Ytrain)
        Ytrain_hat = model.predict(Xtrain)
        print(classification_report(Ytrain, Ytrain_hat,digits=4, output_dict=False, zero_division=0))
        Ytest_hat = model.predict(Xtest)
        if testing_params["evaluate"]:
            print("Evaluating test set...")
            print(classification_report(Ytest, Ytest_hat,digits=4, zero_division=0))
            cr_test = classification_report(Ytest, Ytest_hat,digits=4, output_dict=True, zero_division=0)
            cr_train = classification_report(Ytrain, Ytrain_hat,digits=4, output_dict=True, zero_division=0)
            # ret = pd.DataFrame({"Y": Ytest, "Yhat": Ytest_hat}, index=test_index)
            print(pd.Series(Ytest_hat).value_counts())
            ret=cr_test,Ytest_hat

        if testing_params["return_predict"]:
            ret = pd.DataFrame({"Y":Ytest ,"Yhat":Ytest_hat}, index=test_index)

        return ret


def kfold_train_test(X,Y, index, folds):
    if folds < 1:
        print("Logic error. Should have more than one fold for k-fold validation")

    kf = KFold(n_splits=folds, shuffle=True)
    reports = []
    for i, (kf_train_idx, kf_test_idx) in enumerate(kf.split(X)):
        print("Starting fold", i+1 )
        x_train, y_train = X[kf_train_idx], Y[kf_train_idx]
        x_test, y_test = X[kf_test_idx], Y[kf_test_idx]

        model = LinearSVC().fit(x_train, y_train)
        y_test_hat = model.predict(x_test)
        print(classification_report(y_test, y_test_hat, digits=4, zero_division=0))
        reports.append(classification_report(y_test, y_test_hat, digits=4, output_dict=True, zero_division=0))

    average_classification_reports(reports)

def average_classification_reports(reports:list, ytest_hat=None):
    sum = None
    for i, ret in enumerate(reports):

        inds = [ind for ind in ret.keys()]
        cols = [col for col in ret[inds[0]].keys()]
        inds.remove('accuracy')
        ret = {ind: [ret[ind][col] for col in cols] for ind in inds}
        el = pd.DataFrame.from_dict(ret, orient='index', columns=cols)
        if type(ytest_hat) != None:
            el["#Predictions"] = pd.Series(ytest_hat).value_counts()

        if type(sum) != pd.DataFrame:
            sum = el
        else:
            sum += el
    results = sum / len(reports)
    results = results.drop("support", axis=1)
    print(results)
    print(results.to_latex(float_format="%.3f"))


