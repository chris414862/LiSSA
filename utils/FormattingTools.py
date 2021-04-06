import pandas as pd
import sys
import re
import os
import json

class Preprocessor():

    def __init__(self, anns_fname=None, man_feats_fname=None, docs_fname=None, doc_columns=None
                 , use_man_feat_anns=False, up_to_api_level=None
                 , use_sig_feats=True, ret_only_annotated=True):
        self.anns_fname = anns_fname
        self.man_feats_fname = man_feats_fname
        self.docs_fname = docs_fname
        self.doc_columns = doc_columns
        self.use_man_feats_anns = use_man_feat_anns
        self.up_to_api_level=up_to_api_level
        self.use_sig_feats = use_sig_feats
        self.ret_only_annotated = ret_only_annotated

    def make_index_from_cols(self, df:pd.DataFrame):
        num_recs = df.index.size
        base = df['QualifiedPackage']

        #Prep package and classname
        base = pd.Series(['<']*num_recs).str.cat(base)
        base = base.str.cat(pd.Series(["."]*num_recs))
        base = base.str.cat(df['Classname']).str.cat([': ']*num_recs)

        #Add return type
        ret_types = df["Return"].str.strip()
        ret_types = ret_types.str.split(' ').str.get(0)
        base = base.str.cat(ret_types, na_rep="void").str.cat([' ']*num_recs)

        #Method name
        base = base.str.cat(df["MethodName"]).str.cat(['(']*num_recs)

        #Prep and add parameters
        params = df["Parameters"].str.split(r"\|\|\|", expand=True)
        params = pd.DataFrame({c:params[c].str.split(':').str.get(0).str.split().str.get(1) for c in params.columns})
        params = params.fillna( value="")
        params = params[params.columns].agg(','.join, axis=1).str.rstrip(',')

        # Add ending and remove all "[]"
        # We remove "[]" because some signatures do not include them and we do not want duplicates when we merge dataframes
        base = base.str.cat(params).str.cat([')>']*num_recs)
        base = base.str.replace("[]", repl='', regex=False)
        return base

    def stanardize_index(self, df:pd.DataFrame, is_series=False):
        '''

        :return:
        '''
        # if not is_series:
        idx = pd.Series(df.index).str.replace("[]", repl='', regex=False)
        # else:
        #     idx = df.index.str.replace("[]", repl='', regex=False)
        idx = idx.str.split()
        num_recs = idx.size
        ret_type = idx.str.get(1).str.strip()
        ret_type = ret_type.str.extract("([^.$]+)$")
        classpath = idx.str.get(0).str.extract(r"(.*)\$?.*", expand=False)
        method_name = idx.str.get(2).str.extract(r"(.*)\(", expand=False)
        params = idx.str.get(2).str.extract(r"\((.*)\)", expand=False)
        params = params.str.extractall(r"([^.,$]*),|([^.,$]+$)")
        params = params.fillna(value="")
        params = params.unstack(level=-1, fill_value='')
        params = params.agg(','.join, axis=1).str.strip(',')
        params = params.str.replace(",{2,}", repl=",", regex=True)
        idx = classpath.str.cat([" "] * num_recs)
        idx = idx.str.cat(ret_type)
        idx = idx.str.cat([" "] * num_recs)
        idx = idx.str.cat(method_name)
        idx = idx.str.cat(["("] * num_recs)
        idx = idx.str.cat(params, na_rep="")
        idx = idx.str.cat([")>"] * num_recs)

        return idx

    def formatAnns(self, df:pd.DataFrame):
        df.index = self.stanardize_index(df)

        return df


    def convert_api_level(self, s:pd.Series):
        s = s.str.replace(r"\D+", '', regex=True).fillna(-1)
        s[s == ''] = s[s == ''].replace("", -1)
        return s.astype(int)


    def formatDocs(self, df:pd.DataFrame):
        df.index = self.make_index_from_cols(df)
        df.index = self.stanardize_index(df)
        df["ApiLevel"] = self.convert_api_level(df["ApiLevel"])

        return df


    def formatManFeats(self, df:pd.DataFrame):
        #TODO: uncomment next line
        df.index = self.stanardize_index(df)
        return df


    def replace_missing_with_mode(self, s:pd.Series, missing_label='NOT_SUPPORTED'):
        counts = s.value_counts().sort_values(ascending=False)
        mode = counts.index[0]
        if mode == missing_label:
            mode = counts.index[1]
        s = s.replace(missing_label, mode)
        return s

    def normalize_values(self,s:pd.Series ):
        mean = s.mean()
        std = s.std()

        if(s.sum() <= 0):
            return s
        else:
            new_s = (s- mean)/std
            return new_s

    def convert_str_to_float(self, df):
        #TODO: this function needs to be generalized
        df = df.replace("TRUE", 1.0)
        df = df.replace("FALSE", 0.0)
        df = df.fillna(0.0)
        # df = df.replace("NOT_SUPPORTED", 0.0)
        return df#.astype(dtype=float)


    def extract_df_from_dir(self, dirname:str, filename:str=None):
        '''
        Convertes manual feature data stored in json files in 'dirname' to a single DataFrame and returns it. The
        json files are expected to be the output from the ManualFeatureExtraction tool.
        :param dirname:
        :return:
        '''

        dataframes = []
        for i, filename in enumerate(os.listdir(dirname)):
            if not os.path.isdir(dirname + '/' + filename) and re.search(r"\.json$", filename) is not None:
                print("Loading " + filename + ".....")
                with open(dirname + '/' + filename) as f:
                    data = json.load(f)
                    df = pd.DataFrame.from_dict(data, orient="index")
                dataframes.append(df)
        return pd.concat(dataframes)

    def custom_fill_na(self, df:pd.DataFrame):
        ret_type = df.index.str.split().str.get(1).to_series(index=df.index)
        if 'Return' in df.columns:
            df["Return"] = df['Return'].fillna(ret_type)
        if 'Parameters' in df.columns:
            df["Parameters"] = df["Parameters"].fillna("")
        if 'Description' in df.columns:
            df["Description"] = df["Description"].fillna("")
        if 'Annotation' in df.columns:
            df["Annotation"] = df["Annotation"].fillna("NEITHER")
            df.loc[df['Annotation']=='NONE', 'Annotation'] = 'NEITHER'
        return df


    def preprocess_ann_file(self):
        # use_cols = ['Unnamed: 0', 'Source?', 'Sink?', 'sensitive data?', 'shared resource?']

        print(self.anns_fname)
        df = pd.read_csv(self.anns_fname, index_col=0)#, usecols=use_cols)
        df = self.formatAnns(df)
        return df


    def preprocess_doc_file(self, orig_df):
        names = ['QualifiedPackage', 'Classname', 'MethodName', 'Description', 'Parameters', 'Return', 'ApiLevel']
        docs = pd.read_csv(self.docs_fname, names=names)
        docs = self.formatDocs(docs)
        docs = docs.loc[~docs.index.duplicated(keep='first')]

        if orig_df is not None:
            orig_index = orig_df.index
            orig_df.index = self.stanardize_index(orig_df)
            docs = docs.loc[docs.index.isin(orig_df.index)]
            not_in_docs:pd.DataFrame = orig_df.loc[~orig_df.index.isin(docs.index)].copy()
            if not not_in_docs.empty:
                for col in docs.columns:
                    not_in_docs.loc[:,col] = ''
                not_in_docs = not_in_docs.drop(orig_df.columns, axis=1)
                docs = docs.append(not_in_docs)
            docs = docs.loc[orig_df.index]
            orig_df = pd.concat((orig_df,docs), axis=1)
            orig_df.index = orig_index
        else:
            orig_df = docs

        if self.up_to_api_level is not None:
            print("Removing API methods above level:", self.up_to_api_level)
            print('Original method count:', orig_df.shape[0])
            orig_df = orig_df.loc[orig_df['ApiLevel'] <= self.up_to_api_level]
            print('New method count:', orig_df.shape[0])
        return  orig_df

    def preprocess_man_feats_file(self, orig_df=None):
        '''
        This method returns a the 'orig_df' DataFrame concatenated with the contents of the manual feature
        file name/directory provided to the constructor of the vectorizer. If directory, it should be the directory
        that was output from the ManualFeatureExtraction tool. If file name, it should be a pickled Dataframe from that
        directory.

        :param orig_df:
        :return:
        '''
        if os.path.isdir(self.man_feats_fname):
            man_feats:pd.DataFrame = self.extract_df_from_dir(self.man_feats_fname)
        else:
            print("\n\t"+self.man_feats_fname+" is not a directory.")
            print("\tDetecting file type....")
            mo = re.search(r".*\.pickle", self.man_feats_fname)
            if mo is not None:
                print("\tFound pickled file, attempting to unpickle...", end=" ")
                man_feats:pd.DataFrame = pd.read_pickle(self.man_feats_fname)

            else:
                man_feats:pd.DataFrame = json.load(self.man_feats_fname)

        if self.use_man_feats_anns:
            man_feats_ann = man_feats['AnnotationMF']
            man_feats = man_feats.drop("AnnotationMF", axis=1)
            orig_df = man_feats
            orig_df['Annotation'] = man_feats_ann.values

        if 'AnnotationMF' in man_feats.columns:
            man_feats = man_feats.drop('AnnotationMF', axis=1)

        man_feats = man_feats.loc[~man_feats.index.duplicated(keep='first')]

        # Saving the index bc the manual features' method signatures have qualified types, but
        # we will match the orig_df with unqualified types
        man_orig_index = man_feats.index

        man_feats = self.formatManFeats(man_feats)
        man_feats = man_feats.replace("NOT_SUPPORTED", 0.0)

        man_feats = self.convert_str_to_float(man_feats)
        man_feats = man_feats.astype(float)


        mask = None
        if orig_df is not None:
            orig_df = orig_df.loc[~orig_df.index.duplicated(keep='first')]
            man_feats = self.formatManFeats(man_feats)
            mask = man_feats.index.isin(orig_df.index)
            man_feats = man_feats.loc[mask]
            mask2 = orig_df.index.isin(man_feats.index)
            orig_df2 = orig_df.loc[mask2]

            if man_feats.shape[0] != orig_df.shape[0]:
                print("\nError in aligning manual features to annotations. Shape mismatch in number of methods\n"+
                      "\tNumber of unique annotations:",orig_df.shape[0],"\n\tnumber of unique methods from manual "+
                      "feature extraction:", man_feats.shape[0])

                pd.set_option("display.max_colwidth", 1000)
                print(orig_df.loc[~orig_df.index.isin(man_feats.index)])
                sys.exit()

            orig_df = orig_df.loc[man_feats.index]
            orig_df = pd.concat((orig_df,man_feats), axis=1)
        else:
            orig_df = man_feats

        # Adding the original index back in
        if mask is not None:
            orig_df.index =man_orig_index[mask]

        return orig_df

    def preprocess_data(self)->pd.DataFrame:
        ret_df = None
        # Annotations
        if self.anns_fname is not None:
            print("Processing annotations file...", end=" ")
            ret_df = self.preprocess_ann_file()
            print('Done. Current data dimensions:', ret_df.shape)

        # Manual features file
        if self.man_feats_fname is not None:
            print("Processing manual features file...", end=" ")
            ret_df = self.preprocess_man_feats_file(ret_df)
            print('Done. Current data dimensions:', ret_df.shape)

        # Documentation file
        if self.docs_fname is not None:
            print("Processing documantation file...", end=" ")
            ret_df = self.preprocess_doc_file(ret_df)
            print('Done. Current data dimensions:', ret_df.shape)

        # Add features from method signature
        if self.use_sig_feats:
            print("Extracting signature information...", end=" ")
            self.add_toks_in_signature(ret_df, method_name=True, class_name=True, qual_path=True)
            print('Done. Current data dimensions:', ret_df.shape)

        ret_df = self.custom_fill_na(ret_df)
        ret_df = ret_df.loc[~ret_df.index.duplicated(keep='first')]
        return ret_df

    def split_camel(self,s:pd.Series):
        s = s.str.replace(r"<", repl="", regex=True).str.strip()
        s = s.str.replace(r":", repl="", regex=True).str.strip()
        s = s.str.replace(r"\(.*", repl="", regex=True).str.strip()
        s = s.str.replace(r"\.", repl=" ", regex=True).str.strip()
        s = s.str.replace(r"([^A-Z]*)([A-Z])", repl=r"\1 \2", regex=True).str.lower()
        s = s.str.replace(r" +", repl=" ", regex=True).str.strip()
        return s


    def add_toks_in_signature(self,df,method_name:bool=None, class_name:bool=None, qual_path:bool=None):
        if not (method_name or class_name or qual_path):
            return
        sig = df.index.to_series()
        split_sig = sig.str.split()
        split_cols = []

        if qual_path:
            t =split_sig.str.get(0).str.replace(r"<(.*)\.[A-Za-z]*:", repl=r"\1", regex=True)
            split_cols.append(self.split_camel(t))
        if class_name:
            split_cols.append(self.split_camel(split_sig.str.get(0).str.split(".").str.get(-1)))
        if method_name:
            split_cols.append(self.split_camel(split_sig.str.get(2)))

        for i, col in enumerate(split_cols):
            if i == 0:
                df["SigFeatures"] = col
            else:
                df["SigFeatures"] = df["SigFeatures"].str.cat(col, sep=" ")







if __name__ == "__main__" :
    print("GGG")
    data = {"1":[1,5,4,3],"2":[4,2,7,6], 'Description':['a','b','c','d']}
    index = ["<android.content.ContentResolver: ContentResolver.MimeTypeInfo getTypeInfo(String)>"
             ,"<android.media.session.MediaSessionManager.RemoteUserInfo: int getUid()>"
             ,"<android.telephony.euicc.DownloadableSubscription: String getEncodedActivationCode()>"
             ,"<android.service.notification.NotificationListenerService.Ranking: long getLastAudiblyAlertedMillis()>"]

    df = pd.DataFrame(data, index=index)
    add_toks_in_signature(df, method_name=True, qual_path=True, class_name=True)
    pd.set_option('display.max_colwidth', -1)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    print(df)



