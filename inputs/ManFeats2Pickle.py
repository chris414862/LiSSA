from utils.FormattingTools import Preprocessor
import pandas as pd

man_feat_dir = 'doc_only_man_feats_jsons'
pickled_fname = 'doc_man_feats.pickle'

preprocessor = Preprocessor(man_feats_fname=man_feat_dir)
df = preprocessor.preprocess_man_feats_file()
df.to_pickle(pickled_fname)
