import pandas as pd
import numpy as np
import re
from nltk.tokenize import sent_tokenize,word_tokenize



def make_multi_cols(df:pd.DataFrame):
    multi_cols = []
    for c in df.columns:
        if re.search(r"<.*>", c):
            multi_cols.append(("ManFeatures", c))
        else:
            multi_cols.append(("DocFeatures", c))
    col_index = pd.MultiIndex.from_tuples(multi_cols, names=['first', 'second'])
    df2 = pd.DataFrame(df.to_numpy(), index=df.index, columns=col_index)
    df2["ManFeatures"] = df2["ManFeatures"].fillna(value=False)
    df2["ManFeatures"] = df2["ManFeatures"].astype(float)
    df2["DocFeatures"] = df2["DocFeatures"].fillna("")
    return df2


def vectorize_methods(df:pd.DataFrame, train_levels=None, test_levels=None, use_doc_feats=False, use_ret_feats=False
                      ,use_param_feats=False,task=None, testing_pool=None, use_man_feats=True, simulate_update=False
                      ,use_sig_feats=False, kfold = False):

    df = df.copy()

    if task == "":
        s = df.loc[:,"Source/Sink"]
        s = s.replace(["none", "unannotated",'unknown'], 0)
        s = s.replace("source", 1)
        s = s.replace("sink", 2)
        df.loc[:,"Source/Sink"] = pd.Series(s.astype(float))

    df = make_multi_cols(df)
    random.seed()
    seed= random.randint(0,1000)
    df = df.sample(frac=1.0, random_state=seed)
    testing_source_df = None

    if type(testing_pool) == pd.DataFrame and not testing_pool.empty:
        testing_source_df = make_multi_cols(testing_pool)


    if train_levels is None or not simulate_update:
        if kfold:
            train = df.sample(frac=1.0)

            test = pd.DataFrame({"A":[]})
        else:
            train = df.sample(frac=.7)
            test = df.drop(train.index).sample(frac=1.0)
    elif type(train_levels) != tuple :
       raise ValueError("train_levels should be a tuple")
    else:
        train = df.loc[(df["DocFeatures", "ApiLevel"] >= train_levels[0]) & (df["DocFeatures", "ApiLevel"] <= train_levels[1])].sample(
            frac=1.0)
        if simulate_update and test_levels is not None:
            if not testing_pool.empty:
                test = testing_source_df.loc[(testing_source_df["DocFeatures", "ApiLevel"] >= test_levels[0])
                              & (testing_source_df["DocFeatures", "ApiLevel"] <= test_levels[1])].sample(frac=1.0)

            else:
                print('TEST LEVELS:', test_levels)
                print(df.loc[df["DocFeatures", 'ApiLevel'] > 27].shape)
                test = df.loc[(df["DocFeatures", "ApiLevel"] >= test_levels[0])
                              & (df["DocFeatures", "ApiLevel"] <= test_levels[1])].sample(frac=1.0)

        elif test_levels is None:
            raise ValueError("If train_levels is specified, test_levels must also be given")

        print("Training on API level:", train_levels[0], "through", train_levels[1])
        print("Testing on API level:", test_levels[0], "through", test_levels[1])

    if not kfold:
        print("SET SIZES")
        print("totalset:", (train.index.size+test.index.size))
        print("trainset:", train.index.size)
        print("testset:", test.index.size)

    ind2feat = {}
    feat2ind = {}
    if use_man_feats:
        add_to_mapping(feat2ind, ind2feat, train['ManFeatures'].columns.to_series().to_dict())

        Xtrain = train['ManFeatures'].to_numpy().astype(float)
        if not kfold:
            Xtest = test['ManFeatures'].to_numpy().astype(float)
        else:
            Xtest =None
            Ytest = None
    else:
        print("NO MAN FEATS USED")
        Xtrain = np.zeros(train['ManFeatures'].to_numpy().shape)
        if not kfold:
            Xtest = np.zeros(test['ManFeatures'].to_numpy().shape)
        else:
            Xtest =None
            Ytest = None

    #Convert to scipy sparse matrix
    Xtrain = csr_matrix(Xtrain)
    if not kfold:
        Xtest = csr_matrix(Xtest)
    if use_doc_feats:
        tfidf = TfidfVectorizer(min_df=3, max_df=.8, max_features=20000, ngram_range=(1, 3)).fit(train["DocFeatures", "Description"].tolist())
        add_to_mapping(feat2ind, ind2feat,tfidf.vocabulary_, disambiguator='return_words')
        print("Doc vocab size:",len(tfidf.get_feature_names()))
        XDoctrain = tfidf.transform(train["DocFeatures", "Description"].tolist())
        Xtrain = hstack((Xtrain, XDoctrain)).toarray()
        if not kfold:
            XDoctest = tfidf.transform(test["DocFeatures", "Description"].tolist())
            Xtest = hstack((Xtest, XDoctest)).toarray()
    if use_ret_feats:
        tfidf = TfidfVectorizer(min_df=3, max_df=.8, max_features=2000).fit(train["DocFeatures", "Return"].tolist())
        add_to_mapping(feat2ind, ind2feat,tfidf.vocabulary_, disambiguator='return_words')
        print("Return vocab size:", len(tfidf.get_feature_names()))
        XRettrain = tfidf.transform(train["DocFeatures", "Return"].tolist())
        Xtrain = hstack((Xtrain, XRettrain)).toarray()
        if not kfold:
            XRettest = tfidf.transform(test["DocFeatures", "Return"].tolist())
            Xtest = hstack((Xtest,XRettest)).toarray()

    if use_param_feats:
        tfidf = TfidfVectorizer(min_df=3,max_df=.8,  max_features=2000, ngram_range=(1, 3)).fit(train["DocFeatures", "Parameters"].tolist())
        add_to_mapping(feat2ind, ind2feat,tfidf.vocabulary_, disambiguator='param_words')
        print("Param vocab size:", len(tfidf.get_feature_names()))
        XParamtrain = tfidf.transform(train["DocFeatures", "Parameters"].tolist())
        Xtrain = hstack((Xtrain, XParamtrain)).toarray()
        if not kfold:
            XParamtest = tfidf.transform(test["DocFeatures", "Parameters"].tolist())
            Xtest = hstack((Xtest,XParamtest)).toarray()

    if use_sig_feats:
        tfidf = TfidfVectorizer(min_df=3,max_df=.8,  max_features=8000, ngram_range=(1, 3)).fit(train["DocFeatures", "SigFeatures"].tolist())
        add_to_mapping(feat2ind, ind2feat,tfidf.vocabulary_)
        print("Sig vocab size:", len(tfidf.get_feature_names()))
        XSigtrain = tfidf.transform(train["DocFeatures", "SigFeatures"].tolist())
        Xtrain = hstack((Xtrain, XSigtrain)).toarray()
        if not kfold:
            XSigtest = tfidf.transform(test["DocFeatures", "SigFeatures"].tolist())
            Xtest = hstack((Xtest,XSigtest)).toarray()

    if task == 'source/sink':
        Ytrain = train["DocFeatures", 'Source/Sink'].to_numpy()#.astype(float)
        if not kfold:
            Ytest = test["DocFeatures", 'Source/Sink'].to_numpy()#.astype(float)
    if task == 'category':
        Ytrain = train["DocFeatures", 'Category'].to_numpy()
        if not kfold:
            Ytest = test["DocFeatures", 'Category'].to_numpy()
    if not kfold:
        print("SIZE CHECK: test:",test.index.size,"Ytest",Ytest.shape[0])
    return  (train.index, test.index), (Xtrain, Ytrain, Xtest,Ytest), ind2feat


def extract_target_names(Y, Yhat):
    ret = []
    if Y[Y==0].shape[0] > 0 or Yhat[Yhat==0].shape[0] > 0:
        ret.append("None")
    if Y[Y==1].shape[0] > 0 or Yhat[Yhat==1].shape[0] > 0:
        ret.append("Source")
    if Y[Y==2].shape[0] > 0 or Yhat[Yhat==1].shape[0] > 0:
        ret.append("Sink")
    return ret


def add_to_mapping(label2idx, idx2label, new_dict, disambiguator=''):

    for key,val in new_dict.items():
        if key not in label2idx:
            label2idx[key] = len(label2idx)
            idx2label[len(idx2label)] = key
        else:
            label2idx["<<"+disambiguator+">>"+key] = len(label2idx)
            idx2label[len(idx2label)] = "<<"+disambiguator+">>"+key




