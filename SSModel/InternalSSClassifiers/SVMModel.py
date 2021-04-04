
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from SSModel.ModelInterface import Model
from SSModel.InternalSSVectorizers.BoWVectorizer import BoWVectorizer
import pandas as pd
import numpy as np

class SVMModel(Model):

    def __init__(self):
        self.vectorizer = None
        #Defaults to deterministic behavior. Use reset_w_seed to use stochastically.  
        self.seed = 0
        self.model_class = LinearSVC
        self.model = LinearSVC(random_state=self.seed)
        # self.model = CalibratedClassifierCV(self.svc, 'isotonic',10)

    def get_class_labels(self) -> list:
        return self.model.classes_.tolist()

    def reset_w_seed(self, seed):
        self.seed = seed
        self.model = self.model_class(random_state=seed)


    def get_model_type(self) -> str:
        return 'SVM'

    def get_weights(self):
        ret = self.get_internal_model().coef_
        print("SVM coeficient matrix:", ret.shape)
        return ret


    def get_internal_model(self):
        return self.model

    def train(self, X, y, feat_cols_and_hyperparams=None):
        self.vectorizer = BoWVectorizer()
        # print("X")
        # print(X)
        self.vectorizer.fit_methods(X, feat_cols_and_hyperparams)
        vectorized_X = self.vectorizer.transform_methods(X).todense()
        # print("vectorised_X")
        # print(vectorized_X)
        # print(feat_cols_and_hyperparams)

        self.model.fit(vectorized_X, y)

    def predict(self, X, y=None):
        # scores:np.ndarray = self.model.predict_proba(self.vectorizer.transform_methods(X).todense())
        # print(scores)
        # scores = np.zeros(dec_func.shape)
        # scores[:,0] += np.greater(dec_func[:,0],0).astype(int)
        # scores[:, 1] += np.less(dec_func[:, 0], 0).astype(int)
        # scores[:, 0] += np.greater(dec_func[:, 1], 0).astype(int)
        # scores[:, 2] += np.less(dec_func[:, 1], 0).astype(int)
        # scores[:, 1] += np.greater(dec_func[:, 2], 0).astype(int)
        #
        # scores[:, 2] += np.less(dec_func[:, 2], 0).astype(int)
        # # print(scores)
        # indeces = pd.Series(scores.argmax(axis=1), index=X.index)
        # predict_p = indeces.apply(lambda x: self.model.classes_[x])
        vec_X =self.vectorizer.transform_methods(X).todense()
        predict_reg = pd.Series(self.model.predict(vec_X), index=X.index)
        # if y is not None:
        #     print(roc_auc_score(y,scores, multi_class='ovr', average='weighted'))
        dec_func = pd.DataFrame(self.model.decision_function(vec_X), index=X.index)

        return predict_reg, dec_func#pd.Series(self.model.predict(self.vectorizer.transform_methods(X)), index=X.index)

    def get_vectorizer(self):
        return self.vectorizer
