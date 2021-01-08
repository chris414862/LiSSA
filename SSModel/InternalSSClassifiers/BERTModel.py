import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from SSModel.InternalSSVectorizers.BERTVectorizer import BERTVectorizer
from SSModel.ModelInterface import Model
from SSModel.VectroizerInterface import Vectorizer
from transformers import AdamW
import pandas as pd
from utils.AnnUtils import get_df_from_csv
from tqdm import tqdm
import re
from SSModel.InternalSSVectorizers.BoWVectorizer import BoWVectorizer



class BERTModel(Model):
    def __init__(self, model_class = None, tokenizer_class=None, pretrained_weights=None, num_man_feats=None
                 , trainable_bert_layers:tuple=None):
        self.internal_model = self.BERTInternal( model_class, pretrained_weights, 768, 3, num_man_feats)
        self.vectorizer = BERTVectorizer(tokenizer_class, pretrained_weights)
        self.class_labels = None
        self.model_type = 'BERT'
        def my_filter(x):
            mo = re.search(r"encoder\.layer\.(\d+)\.", x[0])
            if mo is None:
                return True
            try:
                layer_number = int(mo.group(1))
            except ValueError as e:
                print("Namespace conflict:", x[0], "\n'encoder.layer' should be reserved for bert layer.")
                raise e
            if trainable_bert_layers[0] <= layer_number+1 <= trainable_bert_layers[1]:
                return True
            else:
                return False

        if trainable_bert_layers is not None:
            training_params = [p for p in filter(my_filter, self.internal_model.named_parameters())]
        else:
            training_params = self.internal_model.named_parameters()



        self.optimizer = AdamW(training_params)
        self.loss_function = nn.CrossEntropyLoss()

    def get_class_labels(self) -> list:
        return self.class_labels

    def get_model_type(self) -> str:
        return self.model_type

    def get_internal_model(self):
        return self.internal_model

    def get_weights(self):
        return self.internal_model.get_weights()

    def train(self, X:pd.Series, y:pd.Series, batch_size=2, epochs =1, man_feats=None):
        model = self.get_internal_model()
        model.train()
        self.class_labels:list = y.unique().tolist()
        num_entries = y.shape[0]
        for epoch in range(epochs):
            X = X.sample(frac=1.0)
            y = y[X.index]
            y = self.vectorizer.transform_labels(y,labels= self.class_labels)
            with tqdm(total=num_entries) as epoch_pbar:
                epoch_pbar.set_description(f'Epoch {epoch}')
                accum_loss = 0
                for idx, i in enumerate(range(0,len(X), batch_size)):
                    batch_X, batch_y = X[i:i+batch_size], y[i:i+batch_size]
                    batch_man_feats = man_feats[i:i+batch_size]
                    batch_X = self.vectorizer.transform_methods(batch_X)
                    self.optimizer.zero_grad()
                    predictions : torch.Tensor = model.forward(batch_X, batch_man_feats)
                    loss = self.loss_function(predictions, batch_y)
                    loss.backward()
                    self.optimizer.step()

                    # Add loss to accumulated loss
                    accum_loss += loss

                    # Update progress bar description
                    avg_loss = accum_loss / (idx + 1)
                    desc = f'Epoch {epoch} - avg_loss {avg_loss:.4f} - curr_loss {loss:.4f}'
                    epoch_pbar.set_description(desc)
                    epoch_pbar.update(batch_size)


    def get_vectorizer(self) -> Vectorizer:
        raise NotImplementedError()

    def predict(self, X):
        self.vectorizer.transform_methods(X)

    class BERTInternal(nn.Module):

        def __init__(self, model_class, pretrained_weights, embed_dimensions, num_classes, num_man_feats):
            super(BERTModel.BERTInternal, self).__init__()
            self.L1 = model_class.from_pretrained(pretrained_weights)
            self.L2 = self.CustomAttentionLayer(embed_dimensions, num_classes)
            self.final = nn.Linear(embed_dimensions+num_man_feats, num_classes, bias=False)
            self.final_bias = nn.Linear(num_classes, 1, bias=False)
            self.softmax = nn.Softmax(dim=1)



        def forward(self, encoded_input, man_feats:pd.DataFrame):
            input_ids= torch.tensor(encoded_input['input_ids'], dtype=torch.long)
            token_type_ids = torch.tensor(encoded_input['token_type_ids'], dtype=torch.long)
            attention_mask = torch.tensor(encoded_input['attention_mask'], dtype=torch.long)

            # Size of model output ==> (batch_size, seq_len, embed_dimensions)
            model_output, _ = self.L1(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            # S1 size ==> (batch_size, num_classes, embed_dimensions)
            # Each vector in the class dimension represents the document's feature with respect to that class
            S1, word_attention_weights = self.L2.forward(model_output)


            # FINAL LAYER DIMENSION TRACKING
            # output = softmax(sum(haddarmad(S1, W_c), dim=-1)+b_c) ==>
            #   X = haddarmad(S1,W_c):
            #       (batch_size, num_classes, embed_dims + num_manual_features) \hadamard (batch_size, num_classes, embed_dims + num_manual_features)
            #           ==> (batch_size, num_classes, embed_dims + num_manual_features)
            #   X = Sum(X, dim=-1):
            #       \sigma (batch_size, num_classes, embed_dims + num_manual_features) ==> (batch_size, num_classes)
            #   X = X + b_c:
            #       (batch_size, num_classes) + (1, num_classes) ==> (batch_size, num_classes)
            #   softmax(X):
            #       \softmax (batch_size, num_classes) ==> (batch_size, num_classes)

            man_feats_tens = torch.tensor(man_feats.to_numpy(dtype=int), dtype=torch.float32).unsqueeze(dim=1)
            # Manual features are repeated for every class
            man_feats_tens = man_feats_tens.repeat(1,S1.size()[1],1)
            inter = torch.cat((S1,man_feats_tens), dim=-1)

            # Using the Hadamard product and summation ensures there's no interaction between the document's
            # different class representations. This makes analysis more straight forward
            output = self.softmax(torch.sum(torch.mul(inter, self.final.weight), 2, keepdim=False)+self.final_bias.weight)
            return output

        def get_weights(self):
            return None

        class CustomAttentionLayer(nn.Module):
            def __init__(self, dimensions, num_classes):
                super(BERTModel.BERTInternal.CustomAttentionLayer, self).__init__()
                self.linear_in = nn.Linear(dimensions, dimensions)
                self.tanh = nn.Tanh()
                self.queries = nn.Linear(dimensions, num_classes)
                self.softmax = nn.Softmax(dim=2)

            def forward(self, X:torch.Tensor):
                # X.size() == (batch_size, seq_length, embed_dimensions)

                # U = tanh(X*W_w) ==> (batch_size, seq_length, embed_dimensions)*(embed_dimensions, embed_dimensions) -->
                #    (batch_size, seq_length, embed_dimensions)
                U = self.tanh(self.linear_in(X))

                # A = softmax(X*Q +b_q) ==> (batch_size, seq_length, embed_dimensions)*(embed_dimensions, num_classes/queries) -->
                #    (batch_size, seq_length, num_classes/queries)
                attention_weights = self.softmax(self.queries(U))

                # S = A^T*X +b_a (batch_size, num_classes/queries, seq_length)*(batch_size, seq_length, embed_dimensions) -->
                #    (batch_size, num_classes/queries, embed_dimension)
                S = torch.bmm(attention_weights.transpose(1, 2), X)

                return S, attention_weights







if __name__ == "__main__":
    model_class, tokenizer_class, pretrained_weights = BertModel, BertTokenizer, 'bert-base-cased'

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    # model = model_class.from_pretrained(pretrained_weights)

    class_descrip_file = '../../Inputs/class_descriptions_android.csv'
    package_descrip_file = '../../Inputs/package_descriptions_android.csv'
    ignore_if_next_contains = [r'^javax?\..*', r'^com\..*', r'^dalvic\..*', r'^junit\..*', r'^j\..*', r'^junit\..*']
    package_descrip_cols = ['QualPackageName', 'NumMethods', 'Description']
    class_descrip_cols = ['QualClassName', 'NumMethods', 'Description']
    cols_4_class_sig = (0, 2)
    cols_4_package_sig = (0, 1)
    create_cache = False
    cache_name = 'bert_debug_cache.pickle'

    if create_cache:
        df = pd.read_pickle('../../Inputs/Caches/cache2.pickle')
        class_descrips = get_df_from_csv(class_descrip_file, aggregate_cols=cols_4_class_sig, col_names=class_descrip_cols
                                         , ignore_if_next_contains=ignore_if_next_contains, index_col=class_descrip_cols[0])
        package_descrips = get_df_from_csv(package_descrip_file, aggregate_cols=cols_4_package_sig,
                                           col_names=package_descrip_cols
                                           , ignore_if_next_contains=ignore_if_next_contains,
                                           index_col=package_descrip_cols[0], add_period=True)

        cols_to_embed = ['Description', "ClassDescription", "PackageDescription"]
        df["PackageDescription"] = ''
        df['ClassDescription'] = ''
        df_qualified_classname = df['QualifiedPackage'].str.cat( df['Classname'].copy(), sep='.')
        # print(df_qualified_classname)
        for package in package_descrips.index.tolist():
            df.loc[df['QualifiedPackage']== package, 'PackageDescription'] = package_descrips.loc[package, 'Description']
        for classname in class_descrips.index.tolist():
            df.loc[df_qualified_classname== classname, 'ClassDescription'] = class_descrips.loc[classname, 'Description']

        def concat_str_cols(X:pd.DataFrame, columns:list=None):
            combined_data = pd.Series(index=X.index, dtype='object')
            combined_data = combined_data.fillna('')

            for col in columns:
                combined_data= combined_data.str.cat(X[col].copy().fillna(''), sep=' ')
            return combined_data
        s = concat_str_cols(df, cols_to_embed)
        df2 = pd.DataFrame(index=s.index)
        df2['X'] = s.copy()
        df2['y'] = df['Source/Sink'].copy()
        bow = BoWVectorizer()
        mf_cols = bow.find_man_feat_cols(df)
        df2[mf_cols] = df[mf_cols].copy()
        df2.to_pickle(cache_name)
        df = df2
    else:
        print('reading cache')
        df = pd.read_pickle(cache_name)
        bow = BoWVectorizer()
        mf_cols = bow.find_man_feat_cols(df)

    bm = BERTModel(model_class, tokenizer_class, pretrained_weights, len(mf_cols), trainable_bert_layers=(7,12))
    bow = BoWVectorizer()
    mf_cols = bow.find_man_feat_cols(df)

    bm.train(df['X'],df['y'], man_feats = df[mf_cols])


    # for little_s, enc in zip(s[:10],t['input_ids']):
    #     print(re.sub(r"\n", '',little_s))
    #     print(enc)
    #     print(len([e for e in enc if e != 0]))
    # text = df['Description'].to_list()
    # print(text[0])
    # encs = tokenizer.batch_encode_plus(text[:2],add_special_tokens=True, max_length=512, pad_to_max_length=True, return_token_type_ids=True)
    # doc_lens = []
    # input_ids = torch.tensor(encs['input_ids'] , dtype=torch.long)
    # print(input_ids.size())
    # token_type_ids = torch.tensor(encs['token_type_ids'], dtype=torch.long)
    # attention_mask = torch.tensor(encs['attention_mask'], dtype=torch.long)
    # # model = model_class.from_pretrained(pretrained_weights)
    # # last_hidden_state, pooler_output = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    # # print(last_hidden_state.size())
    # custom_bert = BERTModel(pretrained_weights, 768, 512, 3)
    # custom_bert.forward(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    

