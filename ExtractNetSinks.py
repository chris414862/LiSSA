import pandas as pd

SINK_FILE = 'SinksByConfidence.csv'

df = pd.read_csv(SINK_FILE, index_col=0)
df = df.loc[df.index.str.contains(".net.", regex=False)]
pd.set_option("display.max_row", 100)
df:pd.DataFrame = df.loc[df["HypPlaneDist"] > 0]
print(df.shape[0])
df.to_csv("NetSinks.csv")
