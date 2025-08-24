import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

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

# Train XGBoost
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("✅ XGBoost Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------
# Feature Importance
# ---------------------------
import matplotlib.pyplot as plt

importances = model.feature_importances_
features = X.columns

feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

print("\nTop 10 important features (XGBoost):\n", feat_imp.head(10))

plt.figure(figsize=(10,6))
feat_imp.head(10).plot(kind="bar")
plt.title("Top 10 Feature Importances (XGBoost)")
plt.savefig("xgb_feature_importance.png")
plt.close()
print("📊 XGBoost feature importance saved as xgb_feature_importance.png")
