from SSModel.InternalSSClassifiers.SVMModel import SVMModel
from SSModel.ModelInterface import Model
from SSModel.VectroizerInterface import Vectorizer

'''
This class represents the outward-facing interface for general users of the LiSSA model. When instantiating, users
can select the type of internal model to be used. 
'''

class cLiSSAfier():

    def __init__(self, model_type=None):

        if model_type == "SVM":
            self.internal_model : Model =  SVMModel()
        elif model_type == "BERT":
            self.internal_model : Model =  SVMModel()
        else:
            self.internal_model : Model = None

        self.model_type : str = model_type
        self.internal_vectorizer : Vectorizer = self.get_vectorizer()


    def train(self, X, y, **kwargs):
        self.internal_model.train(X, y, **kwargs)

    def predict(self, X):
        return self.internal_model.predict(X)

    def get_vectorizer(self) -> Vectorizer:
        '''
        Returns the Vectorizer object associated with the selected internal model. This can be used to give users
        greater control over evaluation.

        :return: Vectorizer of the internal model of this object
        '''
        return self.internal_model.get_vectorizer() if self.get_internal_model() is not None else None

    def get_internal_model(self) -> Model:
        return self.internal_model


















