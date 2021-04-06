import pandas as pd
from utils.AnnUtils import get_df_from_csv, get_pscout_methods, create_and_save_annotations
from utils.FormattingTools import Preprocessor


'''
This script prepared out annotation sets

Chris Crabtree 1/8/2020

** Reorganized directory structure so relative paths likely invalid
** Chris Crabtee 4/5/2021
'''

cols_to_use = ['Description', 'Parameters', 'Return', 'ApiLevel']
new_cols = ['sensitive data?', 'shared resource?', 'sensitive shared resource?',
            'produces sensitive data from shared resource?', 'writes sensitive data to shared resource?', 'Source?',
            'Sink?']
package_descrip_cols = ['QualPackageName', 'NumMethods', 'Description']
class_descrip_cols = ['QualClassName', 'NumMethods', 'Description']

#Fixes malformed/unwanted package and class description csvs
ignore_if_next_contains = [r'^javax?\..*',r'^com\..*',r'^dalvic\..*',r'^junit\..*',r'^j\..*',r'^junit\..*']
pscout_file = "Inputs/mapping_4.2.2.csv"
class_descrip_file = 'Inputs/class_descriptions_android.csv'
package_descrip_file = 'Inputs/package_descriptions_android.csv'
num_in_subset = 800
android_4_2_outfile = 'to_be_annotated/android_4_2_round_2.csv'
new_api_levels_outfile = 'to_be_annotated/new_apis_round_2.csv'

p = Preprocessor(docs_fname="./Inputs/android_30.csv")
df = p.preprocess_data()
cols_4_class_sig = (0, 2)
cols_4_package_sig = (0, 1)
new_api_levels = (27,29)

# Files that contain methods not to include in new set 
android_4_2_do_not_use_file = './Inputs/FinalizedAnns_4_2_APIs.csv'
new_api_do_not_use_file = './Inputs/FinalizedAnns_NewAPIs.csv'


# Get index of already annotated methods
if android_4_2_do_not_use_file != None:
    android_4_2_dnu_idx = pd.read_csv(android_4_2_do_not_use_file, index_col=0).index
if new_api_do_not_use_file != None:
    new_api_dnu_idx = pd.read_csv(new_api_do_not_use_file, index_col=0).index
    print('new api indices', new_api_dnu_idx)    


class_descrips = get_df_from_csv(class_descrip_file, aggregate_cols=cols_4_class_sig, col_names=class_descrip_cols
                                     , ignore_if_next_contains=ignore_if_next_contains, index_col=class_descrip_cols[0])
package_descrips = get_df_from_csv(package_descrip_file, aggregate_cols=cols_4_package_sig, col_names=package_descrip_cols
                                       , ignore_if_next_contains=ignore_if_next_contains,
                                       index_col=package_descrip_cols[0], add_period=True)
pscout_meths = get_pscout_methods(pscout_file)
print('p meths' , pscout_meths.shape)
print('df shape', df.shape)
pscout_df:pd.DataFrame = df.loc[df.index.isin(pscout_meths)]
print('p df', pscout_df.shape)

# Remove methods that were already annotated
if android_4_2_do_not_use_file != None:
    print('pscout_df', pscout_df)
    print("remove idxs", android_4_2_dnu_idx)
    pscout_df = pscout_df[~pscout_df.index.isin(android_4_2_dnu_idx)]
    print(pscout_df.shape)



# Dont need to specify API level because pscout list is only up to API 18
create_and_save_annotations(pscout_df, num_in_subset,cols_to_use, new_cols, class_descrips,package_descrips
                            ,android_4_2_outfile)

# Create subset of new API methods
new_api_df = df.loc[(new_api_levels[0] <= df['ApiLevel']) & (df["ApiLevel"] <= new_api_levels[1])]
if new_api_do_not_use_file != None:
    new_api_df = new_api_df[~new_api_df.index.isin(new_api_dnu_idx)]
create_and_save_annotations(new_api_df, num_in_subset,cols_to_use, new_cols, class_descrips,package_descrips
                            , new_api_levels_outfile)












