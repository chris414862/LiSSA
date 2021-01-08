import pandas as pd
from utils.SKLearnPrep import vectorize_methods
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import numpy as np


df = pd.read_pickle("../../Inputs/Caches/cache.pickle")
df = df.loc[~df.index.duplicated(keep='first')]
ann_w_doc = df.loc[df["Description"].notna() & (df["Description"] != '') & (df['Origin'] == 'dsafe')]
(train_index, test_index), (Xtrain, Ytrain, Xtest,Ytest), ind2feat = vectorize_methods(ann_w_doc, use_doc_feats=False, use_ret_feats=False
                      ,use_param_feats=False,task= 'source/sink')

model = LinearSVC().fit(Xtrain, Ytrain)

Ytest_hat = model.predict(Xtest)
print(classification_report(Ytest, Ytest_hat,digits=4, output_dict=False, zero_division=0))
sorted_weights = np.argsort( -1*model.coef_, axis=1)
print(sorted_weights.shape)

for i in range(sorted_weights.shape[0]):
    ws = sorted_weights[i,:15].tolist()
    for j in ws:
        print(ind2feat[j], model.coef_[i,j], end='|||')
    print()
print(model.classes_)
