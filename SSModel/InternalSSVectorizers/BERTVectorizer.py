from transformers import BertModel, BertTokenizer
import pandas as pd
from SSModel.VectroizerInterface import Vectorizer
import torch

class BERTVectorizer(Vectorizer):
    def __init__(self, tokenizer_class, pretrained_weights):
        self.tokenizer_class, self.pretrained_weights = tokenizer_class, pretrained_weights
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)


    def fit_methods(self, X:pd.DataFrame):
        ### BERT model has already pretrained its tokenizer and token embeddings
        pass



    def transform_methods(self, X:pd.Series):

        X = X.tolist()
        X = self.tokenizer.batch_encode_plus(X, add_special_tokens=True, max_length=512, pad_to_max_length=True
                                             , return_token_type_ids=True)
        return X
        # input_ids = torch.tensor(X['input_ids'], dtype=torch.long)
        #
        # token_type_ids = torch.tensor(encs['token_type_ids'], dtype=torch.long)
        # attention_mask = torch.tensor(encs['attention_mask'], dtype=torch.long)

    def transform_labels(self, y:pd.Series, labels:list):
        return torch.tensor(y.apply(lambda x: labels.index(x)).astype(int), dtype=torch.long)
