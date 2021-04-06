import pandas as pd

def format_and_save_csv(results_df:pd.DataFrame, info_df, filename:str, sort=True):
    relevent_cols = ['Description', 'Parameters', 'Return', 'ApiLevel']
    for col in relevent_cols:
        results_df[col] = info_df[col][results_df.index]
    # print(results_df.columns)
    if sort:
        results_df.sort_values(by="Yhat", axis=0, ascending=False).to_csv(filename)
    else:
        results_df.to_csv(filename)