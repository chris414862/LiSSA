import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from SSModel.VectroizerInterface import Vectorizer
import re

'''
This class performs one or more BoW vectorizations and concatenates them together and uses pandas DataFrames 
as its the container for data instead of numpy ndarrays. In addition it tracks the feature name (i.e. which word 
corresponds to each index) for the full concatenated transformations.

Chris Crabtree 6/29/2020
'''
class BoWVectorizer(Vectorizer):

    def __init__(self):
        self.feat2idx = {}
        self.idx2feat = {}
        self.man_feat_false_val = False
        self.man_feat_true_val = True
        self.man_feat_cols = None
        self.api_feat_vectorizers = []


    def find_man_feat_cols(self,df:pd.DataFrame):
        '''
        Manual features columns in 'df' should be the only columns to contain '<' and '>'. This method finds and
        stores those columns.

        :param df: Data
        :return: list of manual feature column names
        '''
        man_feat_cols = []
        for c in df.columns:
            # The Manual features columns should contain a description that is surrounded surrounded by '<' and '>'
            if re.search(r"<.*>", c) or c == "Method starting with 'insert' invoked":
                man_feat_cols.append(c)
        return man_feat_cols


    def add_to_mapping(self, new_collection, disambiguator=''):
        '''


        :param new_collection:
        :param disambiguator:
        :return:
        '''
        if type(new_collection) == dict:
            new_collection = [el for el in new_collection.items()]

        for key, val in new_collection:

            self.feat2idx["<<"+disambiguator+">>"+key] = len(self.feat2idx)
            self.idx2feat[len(self.idx2feat)] = "<<"+disambiguator+">>"+key


    def fit_man_feats(self,df:pd.DataFrame):
        '''


        :param df:
        :return:
        '''
        self.man_feat_cols = self.find_man_feat_cols(df)
        print("Manual Features:", len(self.man_feat_cols))
        # print('len df cols:', len(df.columns))
        # print([col for col in df.columns if col not in self.man_feat_cols])
        self.add_to_mapping([(mfc, i) for i, mfc in enumerate(self.man_feat_cols)], 'manual')


    def transform_man_feats(self,df:pd.DataFrame):
        # For some reason the SettingWithCopyWarning exception is thrown by the two next lines despite not having any
        # chained assignments. We decided to just disable the warning
        pd.options.mode.chained_assignment = None

        # Manual features should just be 'on' or 'off' (i.e. 1.0 or 0.0)
        df.loc[:, self.man_feat_cols] = df.loc[:, self.man_feat_cols].fillna(value=0.0)
        df.loc[:,self.man_feat_cols] = df.loc[:,self.man_feat_cols].replace({self.man_feat_true_val:1.0
                                                                            , self.man_feat_false_val:0.0})
        pd.options.mode.chained_assignment = 'warn'

        return df.loc[:,self.man_feat_cols]


    def fit_nlp_features(self, X:pd.Series, hyperparameters:dict=None, col=None):
        tfidf = TfidfVectorizer(min_df=hyperparameters['min_df'], max_df=hyperparameters['max_df']
                                , max_features=hyperparameters['max_features']
                                , ngram_range=hyperparameters['ngram_range']).fit(X.to_numpy())
        self.api_feat_vectorizers.append((X.name, tfidf))
        print(col, len(tfidf.vocabulary_))
        self.add_to_mapping(tfidf.vocabulary_, disambiguator=str(X.name))


    def fit_api_methods(self, df:pd.DataFrame, feature_columns_and_hyperparameters:list=[]):
        for col, hyperparams in feature_columns_and_hyperparameters.items():
            if col == 'Manual_Feats':
                continue

            self.fit_nlp_features(df.loc[:,col], hyperparams, col=col)


    def transform_api_methods(self, df:pd.DataFrame):
        method_mat = None
        for vectorization_tup in self.api_feat_vectorizers:
            col, tfidf = vectorization_tup
            if method_mat is None:
                method_mat = tfidf.transform(df.loc[:,col].to_numpy())
            else:
                method_mat = hstack((method_mat, tfidf.transform(df.loc[:,col].to_numpy())))

        return method_mat


    def fit_methods(self, df:pd.DataFrame, api_cols_and_hyperparams:list=None):
        self.idx2feat, self.feat2idx, self.api_feat_vectorizers = {}, {}, []
        if df is not None:
            if 'Manual_Feats' in api_cols_and_hyperparams and api_cols_and_hyperparams['Manual_Feats']==True:
                self.fit_man_feats(df)
            self.fit_api_methods(df, api_cols_and_hyperparams)


    def transform_methods(self, df: pd.DataFrame):
        if df is not None:
            if self.man_feat_cols is not None:
                man_feats_mat = self.transform_man_feats(df)
            else:
                man_feats_mat = None

            api_feats_mat = self.transform_api_methods(df)
            total_mat = hstack((man_feats_mat, api_feats_mat))
            return total_mat
        else:
            return None




