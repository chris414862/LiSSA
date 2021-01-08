import pandas as pd
from utils.AnnUtils import get_df_from_csv, get_pscout_methods, create_and_save_annotations



cols_to_use = ['Description', 'Parameters', 'Return', 'ApiLevel']
new_cols = ['sensitive data?', 'shared resource?', 'sensitive shared resource?',
            'produces sensitive data from shared resource?', 'writes sensitive data to shared resource?', 'Source?',
            'Sink?']
package_descrip_cols = ['QualPackageName', 'NumMethods', 'Description']
class_descrip_cols = ['QualClassName', 'NumMethods', 'Description']
#Fixes malformed package and class description csvs
ignore_if_next_contains = [r'^javax?\..*',r'^com\..*',r'^dalvic\..*',r'^junit\..*',r'^j\..*',r'^junit\..*']
pscout_file = "Inputs/mapping_4.2.2.csv"
class_descrip_file = 'Inputs/class_descriptions_android.csv'
package_descrip_file = 'Inputs/package_descriptions_android.csv'
num_in_subset = 1000
android_4_2_outfile = 'to_be_annotated/android_4_2_test.csv'
new_api_levels_outfile = 'to_be_annotated/new_apis_test.csv'
df = pd.read_pickle("Inputs/Caches/cache.pickle")
cols_4_class_sig = (0, 2)
cols_4_package_sig = (0, 1)
new_api_levels = (27,29)


class_descrips = get_df_from_csv(class_descrip_file, aggregate_cols=cols_4_class_sig, col_names=class_descrip_cols
                                     , ignore_if_next_contains=ignore_if_next_contains, index_col=class_descrip_cols[0])
package_descrips = get_df_from_csv(package_descrip_file, aggregate_cols=cols_4_package_sig, col_names=package_descrip_cols
                                       , ignore_if_next_contains=ignore_if_next_contains,
                                       index_col=package_descrip_cols[0], add_period=True)
print(package_descrips.index)
pscout_meths = get_pscout_methods(pscout_file)
pscout_df = df.loc[df.index.isin(pscout_meths)]
print(pscout_df.shape)
# Dont need to specify API level because pscout list is only up to API 18
create_and_save_annotations(pscout_df, num_in_subset,cols_to_use, new_cols, class_descrips,package_descrips
                            ,android_4_2_outfile)
new_api_df = df.loc[(new_api_levels[0] <= df['ApiLevel']) & (df["ApiLevel"] <= new_api_levels[1])]
create_and_save_annotations(new_api_df, num_in_subset,cols_to_use, new_cols, class_descrips,package_descrips
                            , new_api_levels_outfile)












