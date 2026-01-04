import pandas as pd

# Correct dataset path
DATA_PATH = "data/text/data/train.tsv"

# Load TSV correctly (NO header in file)
df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["text", "labels", "id"]
)

print("Total samples:", len(df))
print("Columns:", df.columns.tolist())

print("\nSample text:")
print(df.iloc[0]["text"])

print("\nSample labels:")
print(df.iloc[0]["labels"])
