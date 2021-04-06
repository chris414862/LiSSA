import pandas as pd

annotator1_files = ['chris_new_apis_sample.csv', 'chris_android_4_2_FIXED.csv']
annotator2_files = ['rocky_new_apis_sample.csv', 'rocky_android_4_2_FIXED.csv']
save_file_names = ['new_apis_KAPPA.csv', 'android_4_2_KAPPA.csv']
annotator1_name = 'Chris'
annotator2_name = 'Rocky'
annotation_cols = ['Source?', 'Sink?']

def prune_and_merge_cols(df, name):
    df = df[annotation_cols]
    df = df.fillna('no')
    df = df.replace('no', 'NEITHER')
    df[name] = 'NEITHER'
    df.loc[df[annotation_cols[0]] == 'SOURCE', name] = 'SOURCE'

    df.loc[df[annotation_cols[1]] == 'SINK', name] = 'SINK'
    return df


for annotations in zip(annotator1_files, annotator2_files, save_file_names):
    df1, df2 = pd.read_csv(annotations[0], index_col=0), pd.read_csv(annotations[1], index_col=0)
    df1, df2 = prune_and_merge_cols(df1, annotator1_name),prune_and_merge_cols(df2, annotator2_name)
    df = pd.DataFrame(index=df1.index)
    df[annotator1_name], df[annotator2_name] = df1[annotator1_name], df2[annotator2_name]

    df['Agree'] = df['Chris'] == df['Rocky']
    print(annotations[2])
    print(df['Agree'].value_counts())

    # print(df)
    # print(annotations[2])
    # print(df['Chris'].value_counts())
    # df.to_csv(annotations[2])



