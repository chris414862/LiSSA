import pandas as pd
import re
from utils.AnnUtils import get_df_from_csv, create_annotations, get_pscout_methods


messed_up_ann_file = 'Annotations/rocky_android_4_2_sample.csv'
appended_ann_file = 'Annotations/chris_android_4_2_sample_APPENDED.csv'#None#re.sub(r"(.*)\.csv", r"\1_APPENDED.csv", messed_up_ann_file)#'to_be_annotated/android_4_2_sample_APPENDED.csv'
file2save_appended = 'Annotations/chris_android_4_2_sample_APPENDED.csv'
fixed_ann_file = re.sub(r"(.*)\.csv", r"\1_FIXED.csv", messed_up_ann_file)
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
android_4_2_outfile = 'to_be_annotated/android_4_2_test.csv'
new_api_levels_outfile = 'to_be_annotated/new_apis_test.csv'
messed_up_ann_df = pd.read_pickle("Inputs/Caches/cache.pickle")
cols_4_class_sig = (0, 2)
cols_4_package_sig = (0, 1)
new_api_levels = (28,29)
check_is_in_pscout = False





messed_up_ann_df = pd.read_csv(messed_up_ann_file, index_col=0)
desired_anns = messed_up_ann_df.shape[0]
messed_up_ann_df = messed_up_ann_df.loc[~messed_up_ann_df.index.duplicated(keep='first')]
num_dup_anns = desired_anns - messed_up_ann_df.shape[0]
print("Number of duplicated annotations:", num_dup_anns)
df_big = pd.read_pickle("Inputs/Caches/cache.pickle")

class_descrips = get_df_from_csv(class_descrip_file, aggregate_cols=cols_4_class_sig, col_names=class_descrip_cols
                                     , ignore_if_next_contains=ignore_if_next_contains, index_col=class_descrip_cols[0])
package_descrips = get_df_from_csv(package_descrip_file, aggregate_cols=cols_4_package_sig, col_names=package_descrip_cols
                                       , ignore_if_next_contains=ignore_if_next_contains,
                                       index_col=package_descrip_cols[0], add_period=True)



if appended_ann_file is None:
    if check_is_in_pscout:
        pscout_meths = get_pscout_methods(pscout_file)
        pscout_df = df_big.loc[df_big.index.isin(pscout_meths)]
        pscout_df = pscout_df.drop(messed_up_ann_df.index)
        df_to_replace_from = pscout_df
    else:
        df_to_replace_from = df_big.drop(messed_up_ann_df.index)
        print(df_to_replace_from.columns)
        df_to_replace_from = df_to_replace_from.loc[(df_to_replace_from['ApiLevel'] >= new_api_levels[0]) &
                                                    (df_to_replace_from['ApiLevel'] >= new_api_levels[1])]
    df_2 = create_annotations(df_to_replace_from, num_dup_anns, cols_to_use, new_cols, class_descrips, package_descrips
                          , android_4_2_outfile)
else:

    df_2 = pd.read_csv(appended_ann_file, index_col=0)


messed_up_ann_df = messed_up_ann_df.append(df_2)

if file2save_appended is not None:
    df_2.to_csv(file2save_appended)
if fixed_ann_file is not None:
    messed_up_ann_df.to_csv(fixed_ann_file)



