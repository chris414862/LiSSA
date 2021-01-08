import sys
import pandas as pd
from sklearn.metrics import classification_report



def reformat_labels(df):
    df["Y(0=source, 1=sink, 2=none)"] = df["Y(0=source, 1=sink, 2=none)"].replace(0,'source')
    df["Y(0=source, 1=sink, 2=none)"] = df["Y(0=source, 1=sink, 2=none)"].replace(1,'sink')
    df["Y(0=source, 1=sink, 2=none)"] = df["Y(0=source, 1=sink, 2=none)"].replace(2,'none')
    df["Y"] = df["Y(0=source, 1=sink, 2=none)"]
    df = df.drop("Y(0=source, 1=sink, 2=none)", axis=1)


if __name__ == "__main__":
    annotated_df = pd.read_csv(sys.argv[1])
    hypothesis_df = pd.read_csv(sys.argv[2])
    reformat_labels(annotated_df)
    print(classification_report(annotated_df["Y"], hypothesis_df["Yhat"], digits=4, zero_division=0))