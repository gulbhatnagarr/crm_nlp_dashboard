import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preprocess
df = df.drop("customerID", axis=1)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df = df.replace({"Yes": 1, "No": 0})
df = pd.get_dummies(df, drop_first=True)

# Features & target
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("✅ Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------
# Feature Importance
# ---------------------------
importances = model.feature_importances_
features = X.columns

# Sort features by importance
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

print("\nTop 10 important features:\n", feat_imp.head(10))

# Plot feature importance
plt.figure(figsize=(10,6))
feat_imp.head(10).plot(kind="bar")
plt.title("Top 10 Feature Importances")
plt.show()