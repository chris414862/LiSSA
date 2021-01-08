from SSModel.VectroizerInterface import Vectorizer

class Model():

    def get_class_labels(self) -> list:
        raise NotImplementedError()

    def get_model_type(self) -> str:
        raise NotImplementedError()

    def get_internal_model(self):
        raise NotImplementedError()

    def get_weights(self):
        raise NotImplementedError()

    def train(self, X, y, **kwargs):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

    def get_vectorizer(self) -> Vectorizer:
        raise NotImplementedError()