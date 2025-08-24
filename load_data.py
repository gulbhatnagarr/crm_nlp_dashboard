import pandas as pd

# Load the dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Basic info
print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())