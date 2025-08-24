import pandas as pd

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Before cleaning:", df.shape)

# 1. Drop customerID (not useful for ML)
df = df.drop("customerID", axis=1)

# 2. Convert TotalCharges to numeric (some values are blank)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# 3. Handle missing values
df = df.dropna()

# 4. Convert Yes/No to 1/0
df = df.replace({"Yes": 1, "No": 0})

# 5. One-hot encode categorical features
df = pd.get_dummies(df, drop_first=True)

print("After cleaning:", df.shape)
print("Columns after preprocessing:", df.columns.tolist()[:10], "...")  # show first 10 columns