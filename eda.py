import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# ----------------------------
# 1. Churn distribution
# ----------------------------
plt.figure(figsize=(6,6))
df["Churn"].value_counts().plot(kind="pie", autopct="%1.1f%%", colors=["lightgreen","salmon"])
plt.title("Churn Distribution")
plt.ylabel("")
plt.savefig("eda_churn_distribution.png")
plt.close()
print("📊 Saved churn distribution pie chart as eda_churn_distribution.png")

# ----------------------------
# 2. Tenure vs Churn
# ----------------------------
plt.figure(figsize=(8,6))
sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", bins=30, palette="Set2")
plt.title("Customer Tenure vs Churn")
plt.savefig("eda_tenure_vs_churn.png")
plt.close()
print("📊 Saved tenure vs churn histogram as eda_tenure_vs_churn.png")

# ----------------------------
# 3. MonthlyCharges vs Churn
# ----------------------------
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", palette="Set2")
plt.title("Monthly Charges vs Churn")
plt.savefig("eda_monthlycharges_vs_churn.png")
plt.close()
print("📊 Saved monthly charges vs churn boxplot as eda_monthlycharges_vs_churn.png")

# ----------------------------
# 4. Contract Type vs Churn
# ----------------------------
plt.figure(figsize=(8,6))
sns.countplot(data=df, x="Contract", hue="Churn", palette="Set2")
plt.title("Contract Type vs Churn")
plt.savefig("eda_contract_vs_churn.png")
plt.close()
print("📊 Saved contract type vs churn chart as eda_contract_vs_churn.png")