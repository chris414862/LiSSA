import pandas as pd

'''
This is an interface for classes that perform vectorization for a 'Model' class. Each 'Model' type will have 
individual repuirements/expectations for the vectors representing a method. This interface allows for flexibility
in the implementation to meet these requirements. This class contains the methods that the 'Model' class expects and 
therefore must be implemented by any class that uses this interface.
'''
class Vectorizer():

    def fit_methods(self, X:pd.DataFrame, **kwargs):
        '''
        Trains the vectorization model. Many types of vectorization models require a training set in order to exract
        necessary parameters to transform data in a uniform manner. This method should perform the necessary steps to
        accomplish that when implemented.

        :param args:
        :param kwargs:
        :return:
        '''
        raise NotImplementedError()


    def transform_methods(self, X:pd.DataFrame, **kwargs):
        '''
        Convert the ...

        :param args:
        :param kwargs:
        :return:
        '''
        raise NotImplementedError()

